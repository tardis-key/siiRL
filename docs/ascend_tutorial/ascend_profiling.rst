在昇腾设备上基于FSDP后端进行数据采集
====================================

Last updated: 08/14/2025.

这是一份在昇腾设备上基于FSDP后端使用GRPO进行数据采集的教程。

配置
----

- 全局采集控制:使用siirl/client/config/ppo_trainer.yaml中的配置项控制采集的默认模式

通过 ppo_trainer.yaml 中的参数控制采集参数：

- enable: 是否启用性能分析。
- save_path: 保存采集数据的路径。

- level: 采集级别—选项有 level_none、level0、level1 和 level2
   -  level_none: 禁用所有基于级别的数据采集（关闭 profiler_level）。
   -  level0: 采集高级应用数据、底层NPU数据和NPU上的算子执行详情。
   -  level1: 在level0基础上增加CANN层AscendCL数据和NPU上的AI Core性能指标。
   -  level2: 在level1基础上增加CANN层Runtime数据和AI CPU指标。

- with memory: 是否启用内存分析（默认为True）。
- record shapes: 是否记录张量形状(默认为False)。
- with npu: 是否采集设备端性能数据(默认为True)。
- with cpu: 是否采集主机端性能数据(默认为True)。
- with module: 是否记录框架层Python调用栈信息。
- with stack: 是否记录算子调用栈信息。
- analysis: 启用自动数据解析。
- discrete: 是否启用离散模式，分别收集各个阶段的性能数据(默认为False)

- roles: 采集阶段-与discrete参数配合使用-选项有
    generate,compute_reward,compute_old_log_prob,compute_ref_log_prob,compute_value,compute_advantage,
    train_critic,train_actor

- all_ranks: 是否从所有rank收集数据。
- ranks: 要收集数据的rank列表。如果为空，则不收集数据。
- profile_steps: 采集步数的列表，例如 [2, 4]，表示将采集第2步和第4步。如果设置为 null，则不进行采集。

示例
----

禁用采集
~~~~~~~~~~~~~~~~~~~~
.. code:: yaml

    profiler:
         enable: False # disable profile

端到端采集
~~~~~~~~~~~~~~~~~~~~~

.. code:: yaml

    profiler:
        steps: [1, 2, 5]
        discrete: False

在examples/grpo_trainer下提供了run_qwen2_5-7b-npu-e2e_prof.sh提供端到端采集脚本参考

离散模式采集
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: yaml

    profiler:
        discrete: True
        roles:['generate','train_actor']

在examples/grpo_trainer下提供了run_qwen2_5-7b-npu-discrete_prof.sh提供离散模式采集脚本参考

可视化
------

采集后的数据存放在用户设置的save_path下，可通过 `MindStudio Insight <https://www.hiascend.com/document/detail/zh/mindstudio/80RC1/GUI_baseddevelopmenttool/msascendinsightug/Insight_userguide_0002.html>`_ 工具进行可视化。

如果analysis参数设置为False，采集之后需要进行离线解析：
.. code:: python
    import argparse
    from torch_npu.profiler.profiler import analyse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="facebook/opt-125m")

    if __name__ == "__main__":
    args = parser.parse_args()
    path = args.path

    analyse(profiler_path=path)
