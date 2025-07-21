# Copyright 2025, Shanghai Innovation Institute. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch.distributed as dist
from loguru import logger

from siirl.utils.metrics.metric_utils import aggregate_validation_metrics
from siirl.workers.dag.node import NodeRole
from siirl.workers.dag_worker.data_structures import ValidationPayload, ValidationResult
from siirl.workers.databuffer import DataProto, pad_dataproto_to_divisor, unpad_dataproto


class ValidationMixin:
    """Handles the validation process, including generation, scoring, and aggregation."""

    from typing import Any, Dict, List, Optional
    from collections import defaultdict
    import torch.distributed as dist
    from siirl.utils.params import SiiRLArguments
    from siirl.dataloader import DataLoaderNode
    from siirl.workers.dag.node import Node, NodeRole
    from siirl.workers.base_worker import Worker

    timers: defaultdict
    _rank: int
    global_steps: int
    logger: Any  # Can be more specific if Tracking class is imported
    dataloader: DataLoaderNode
    _gather_group: Optional[dist.ProcessGroup]
    world_size: int  # Assuming this is available, though it's not a self attribute in DAGWorker
    config: SiiRLArguments
    agent_group_worker: Dict[int, Dict[NodeRole, Worker]]
    validate_tokenizer: Any
    first_rollout_node: Node
    val_reward_fn: Any

    _timer: Any
    _get_node_dp_info: Any

    def _validate(self) -> Dict[str, float]:
        """Performs validation by generating, scoring, and aggregating metrics across all ranks."""
        self.timers = defaultdict(float)
        if self._rank == 0:
            logger.info("=" * 60)
            logger.info(f"Starting Validation @ Global Step {self.global_steps}...")
            logger.info("=" * 60)
            self.timers["overall_start_time"] = time.perf_counter()

        all_results_payloads: List[ValidationPayload] = []

        # Check if num_val_batches > 0 to avoid unnecessary loops.
        if self.dataloader.num_val_batches <= 0:
            if self._rank == 0:
                logger.warning("num_val_batches is 0. Skipping validation.")
            return {}

        for i in range(self.dataloader.num_val_batches):
            if self._rank == 0:
                logger.debug(f"Processing validation batch {i + 1}/{self.dataloader.num_val_batches}")

            # Each rank performs generation and scoring on its slice of data
            with self._timer("prep_and_generate", self.timers):
                batch_proto = self._prepare_validation_batch()
                generated_proto = self._generate_for_validation(batch_proto)
                dist.barrier(self._gather_group)  # Ensure generation is complete on all ranks

            with self._timer("score_and_package", self.timers):
                scored_results = self._score_and_package_results(generated_proto)
                payloads = [ValidationPayload(r.input_text, r.score, r.data_source, r.extra_rewards) for r in scored_results]

            all_results_payloads.extend(payloads)

        # Gather all lightweight payloads to rank 0
        dist.barrier(self._gather_group)
        with self._timer("gather_payloads", self.timers):
            gathered_payloads_on_rank0 = [None] * self.world_size if self._rank == 0 else None
            dist.gather_object(all_results_payloads, gathered_payloads_on_rank0, dst=0, group=self._gather_group)

        # Rank 0 performs the final aggregation and logging
        if self._rank == 0:
            flat_payload_list = [p for sublist in gathered_payloads_on_rank0 if sublist for p in sublist]
            return self._aggregate_and_log_validation_metrics(flat_payload_list)

        return {}  # Non-zero ranks return an empty dict

    def _prepare_validation_batch(self) -> DataProto:
        """Fetches and prepares a single batch for validation."""
        test_batch = self.dataloader.run(is_validation_step=True)
        test_batch_proto = DataProto.from_single_dict(test_batch)
        n_samples = self.config.actor_rollout_ref.rollout.val_kwargs.n
        return test_batch_proto.repeat(n_samples, interleave=True)

    def _prepare_generation_batch(self, batch: DataProto) -> DataProto:
        """Pops keys from a batch to isolate data needed for sequence generation."""
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
        if "multi_modal_inputs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.extend(["multi_modal_data", "multi_modal_inputs"])
        if "tools_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("tools_kwargs")
        return batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )

    def _generate_for_validation(self, batch_proto: DataProto) -> DataProto:
        """Generates sequences using the rollout worker for a validation batch."""
        rollout_worker = self.agent_group_worker[0][NodeRole.ROLLOUT]
        val_kwargs = self.config.actor_rollout_ref.rollout.val_kwargs
        gen_batch = self._prepare_generation_batch(batch_proto)

        gen_batch.meta_info = {
            "eos_token_id": self.validate_tokenizer.eos_token_id,
            "pad_token_id": self.validate_tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": val_kwargs.do_sample,
            "validate": True,
        }

        dp_size, _, _, _ = self._get_node_dp_info(self.first_rollout_node)
        gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, dp_size)
        output_padded = rollout_worker.generate_sequences(gen_batch_padded)
        output_unpadded = unpad_dataproto(output_padded, pad_size=pad_size)
        return batch_proto.union(output_unpadded)

    def _score_and_package_results(self, generated_proto: DataProto) -> List[ValidationResult]:
        """Scores generated sequences and packages them into ValidationResult objects."""
        reward_result = self.val_reward_fn(generated_proto, return_dict=True)
        scores = reward_result["reward_tensor"].sum(-1).cpu()

        input_texts = self.validate_tokenizer.batch_decode(generated_proto.batch["input_ids"], skip_special_tokens=True)
        output_texts = self.validate_tokenizer.batch_decode(generated_proto.batch["responses"], skip_special_tokens=True)
        data_sources = generated_proto.non_tensor_batch.get("data_source", ["unknown"] * len(scores))

        packaged_results = []
        for i in range(len(scores)):
            extra_rewards = {k: v[i] for k, v in reward_result.get("reward_extra_info", {}).items()}
            packaged_results.append(ValidationResult(input_texts[i], output_texts[i], scores[i].item(), data_sources[i], reward_result["reward_tensor"][i], extra_rewards))
        return packaged_results

    def _aggregate_and_log_validation_metrics(self, all_payloads: List[ValidationPayload]) -> Dict[str, float]:
        """On Rank 0, aggregates all validation results and logs performance."""
        if not all_payloads:
            logger.warning("Validation finished with no results gathered on Rank 0 to aggregate.")
            return {}

        logger.info(f"Rank 0: Aggregating {len(all_payloads)} validation results...")
        with self._timer("final_aggregation", self.timers):
            final_metrics = self._aggregate_validation_results(all_payloads)

        # Log performance breakdown
        total_time = time.perf_counter() - self.timers.pop("overall_start_time", time.perf_counter())
        logger.info("--- Validation Performance Breakdown (Rank 0) ---")
        for name, duration in self.timers.items():
            logger.info(f"  Total {name.replace('_', ' ').title():<25}: {duration:.4f}s")
        known_time = sum(self.timers.values())
        logger.info(f"  {'Other/Overhead':<25}: {max(0, total_time - known_time):.4f}s")
        logger.info(f"  {'TOTAL VALIDATION TIME':<25}: {total_time:.4f}s")
        logger.info("=" * 51)

        return final_metrics

    def _aggregate_validation_results(self, all_payloads: List[ValidationPayload]) -> Dict[str, float]:
        """Computes the final metric dictionary from all gathered validation payloads."""
        data_sources = [p.data_source for p in all_payloads]
        sample_inputs = [p.input_text for p in all_payloads]

        infos_dict = defaultdict(list)
        for p in all_payloads:
            infos_dict["reward"].append(p.score)
            for key, value in p.extra_rewards.items():
                infos_dict[key].append(value)

        data_src2var2metric2val = aggregate_validation_metrics(data_sources=data_sources, sample_inputs=sample_inputs, infos_dict=infos_dict)

        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                if not metric2val:
                    continue

                # Robustly parse '@N' to prevent crashes from malformed metric names.
                n_max_values = []
                for name in metric2val.keys():
                    if "@" in name and "/mean" in name:
                        try:
                            n_val = int(name.split("@")[-1].split("/")[0])
                            n_max_values.append(n_val)
                        except (ValueError, IndexError):
                            continue  # Ignore malformed metric names

                n_max = max(n_max_values) if n_max_values else 1

                for metric_name, metric_val in metric2val.items():
                    is_core_metric = (var_name == core_var) and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"]) and (f"@{n_max}" in metric_name)

                    metric_sec = "val-core" if is_core_metric else "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        # Re-calculate test_score per data source

        data_source_rewards = defaultdict(list)
        for p in all_payloads:
            data_source_rewards[p.data_source].append(p.score)

        for source, rewards in data_source_rewards.items():
            if rewards:
                metric_dict[f"val/test_score/{source}"] = np.mean(rewards)

        return metric_dict
