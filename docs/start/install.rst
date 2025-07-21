Installation
============

Requirements
------------

- **Python**: Version >= 3.10
- **CUDA**: Version >= 12.1

Currently, siiRL supports the following configurations are available:

- **FSDP** for training.
- **SGLang** and **vLLM** for rollout generation.

Install from docker image
-------------------------

The stable image is ``siiai/siirl-base:vllm0.8.5.post1-sglang0.4.6.post5-cu124``. This images contains the latest version of inference and training framework and its dependencies.

Install from custom environment
---------------------------------------------

We recommend to use docker images for convenience. However, if your environment is not compatible with the docker image, you can also install siirl in a python environment.


Pre-requisites
::::::::::::::

For training and inference engines to utilize better and faster hardware support, CUDA/cuDNN and other dependencies are required,
and some of the dependencies are easy to be overridden when installing other packages,
so we put them in the :ref:`Post-installation` step.

.. note::

    The installation steps below are recommended configurations for the latest version of siirl.
    If you are trying to customize your own environment, please ignore the strict constraints.

We need to install the following pre-requisites:

- **CUDA**: Version >= 12.4
- **cuDNN**: Version >= 9.8.0
- **Apex**

CUDA above 12.4 is recommended to use as the docker image,
please refer to `NVIDIA's official website <https://developer.nvidia.com/cuda-toolkit-archive>`_ for other version of CUDA.

.. code:: bash

    # change directory to anywher you like, in siirl source code directory is not recommended
    wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda-repo-ubuntu2204-12-4-local_12.4.1-550.54.15-1_amd64.deb
    dpkg -i cuda-repo-ubuntu2204-12-4-local_12.4.1-550.54.15-1_amd64.deb
    cp /var/cuda-repo-ubuntu2204-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/
    apt-get update
    apt-get -y install cuda-toolkit-12-4
    update-alternatives --set cuda /usr/local/cuda-12.4


cuDNN can be installed via the following command,
please refer to `NVIDIA's official website <https://developer.nvidia.com/rdp/cudnn-archive>`_ for other version of cuDNN.

.. code:: bash

    # change directory to anywher you like, in siirl source code directory is not recommended
    wget https://developer.download.nvidia.com/compute/cudnn/9.8.0/local_installers/cudnn-local-repo-ubuntu2204-9.8.0_1.0-1_amd64.deb
    dpkg -i cudnn-local-repo-ubuntu2204-9.8.0_1.0-1_amd64.deb
    cp /var/cudnn-local-repo-ubuntu2204-9.8.0/cudnn-*-keyring.gpg /usr/share/keyrings/
    apt-get update
    apt-get -y install cudnn-cuda-12

NVIDIA Apex is required for FSDP training.
You can install it via the following command, but notice that this steps can take a very long time.
It is recommended to set the ``MAX_JOBS`` environment variable to accelerate the installation process,
but do not set it too large, otherwise the memory will be overloaded and your machines may hang.

.. code:: bash

    # change directory to anywher you like, in siirl source code directory is not recommended
    git clone https://github.com/NVIDIA/apex.git && \
    cd apex && \
    MAX_JOB=32 pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./


Install dependencies
::::::::::::::::::::

.. note::

    We recommend to use a fresh new conda environment to install siirl and its dependencies.

    **Notice that the inference frameworks often strictly limit your pytorch version and will directly override your installed pytorch if not paying enough attention.**

    As a countermeasure, it is recommended to install inference frameworks first with the pytorch they needed. For vLLM, if you hope to use your existing pytorch,
    please follow their official instructions
    `Use an existing PyTorch installation <https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html#build-wheel-from-source>`_ .


1. First of all, to manage environment, we recommend using conda:

.. code:: bash

   conda create -n siirl python==3.10
   conda activate siirl

2. Install python packages

.. code:: bash

    pip install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.6/
    pip install flash-attn==2.7.3 --no-build-isolation
    pip install accelerate codetiming datasets dill hydra-core pandas wandb loguru tensorboard qwen_vl_utils
    pip install 'ray[default]>=2.47.1'
    pip install opentelemetry-exporter-prometheus==0.47b0


3. Then, execute the following commands to install vLLM and SGLang:

.. code:: bash

    pip install vllm==0.8.5.post1
    pip install 'sglang[all]==0.4.6.post5'


Install siirl
::::::::::::::

For installing the latest version of siirl, the best way is to clone and
install it from source. Then you can modify our code to customize your
own post-training jobs.

.. code:: bash

   git clone https://github.com/sii-research/siiRL.git
   cd siirl
   pip install -e .


Post-installation
:::::::::::::::::

Please make sure that the installed packages are not overridden during the installation of other packages.

The packages worth checking are:

- **torch** and torch series
- **vLLM**
- **SGLang**
- **pyarrow**
- **tensordict**
- **nvidia-cudnn-cu12**

If you encounter issues about package versions during running siirl, please update the outdated ones.
