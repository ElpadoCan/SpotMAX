.. _Cell-ACDC: https://cell-acdc.readthedocs.io/en/latest/index.html
.. _Environments: https://github.com/ElpadoCan/SpotMAX/tree/main/envs
.. _pyenv: https://github.com/pyenv/pyenv
.. _Install Miniconda: https://docs.anaconda.com/free/miniconda/#quick-command-line-install
.. _Install PyTorch: https://pytorch.org/get-started/locally/
.. _3D model: https://hmgubox2.helmholtz-muenchen.de/index.php/s/eoeFcgsAMDsgTgw
.. _2D model: https://hmgubox2.helmholtz-muenchen.de/index.php/s/4dxeHSLDfAbC8dA
.. _Install CUDA drivers: https://developer.nvidia.com/cuda-downloads
.. _de.NBI cloud: https://cloud.denbi.de/


.. _install-on-hpc:

Install on HPC cluster
----------------------

HPC cluster often do not have a desktop environment, meaning that you need to 
install the headless version of SpotMAX. 

Since most of the HPC clusters run on some Linux-based OS, we recommend using 
``conda`` not only to manage the environments, but also to install the 
dependencies. 

That means installing all the dependencies **first** and then install `Cell-ACDC`_ 
and SpotMAX **without dependencies**. 

In the `Environments`_ folder, you will find the following files:

* ``conda_env_headless.yml`` to install dependencies with ``conda``.
* ``requirements_headless.txt`` to install dependencies with ``pip``. Note that 
  ``pip`` can also be used within a ``conda`` environment.

Follow these steps to install SpotMAX headless:

1. **Copy environment file(s)**
   
   Copy the files above on a folder on the cluster or download them automatically 
   from the terminal with the following commands::

    curl -O https://github.com/ElpadoCan/SpotMAX/blob/main/envs/conda_env_headless.yml
    curl -O https://github.com/ElpadoCan/SpotMAX/blob/main/envs/requirements_headless.txt

2. **Install the package manager**
   
   .. tabs:: 

        .. tab:: conda

            If ``conda`` is not already installed on the cluster, install 
            Miniconda by following this guide `Install Miniconda`_.
        
        .. tab:: pip

            Pip is installed by installing Python. If Python is not already 
            installed on the cluster, we recommend using `pyenv`_ to manage 
            Python installation. 

3. **Create environment and install dependencies**
   
   Navigate in the terminal to the folder where you downloaded the environment 
   files and run the following command(s):

   .. tabs:: 

        .. tab:: conda

            .. code-block:: 
   
                conda env create -f conda_env_headless.yml
        
        .. tab:: pip

            .. code-block:: 
                
                python3 -m venv <path_to_env>\acdc
                source <path_to_env>\acdc\Scripts\activate
                python3 -m pip install -r requirements_headless.txt

4. **Install additional dependencies from pip**:
   
   Some of SpotMAX dependencies are available only from ``pip`` which means 
   you need to install them manually with the following commands::

    pip install --no-deps "git+https://github.com/SchmollerLab/Cell_ACDC.git"
    pip install "git+https://github.com/ElpadoCan/pytorch3dunet.git"

5. **Install SpotMAX from pip**:
   
   Install SpotMAX with the following command::

    pip install "git+https://github.com/ElpadoCan/SpotMAX.git"

6. **Install PyTorch** (optional):

   To install PyTorch follow this guide `Install PyTorch`_. If you have an 
   NVIDIA GPU and a CUDA-capable system, make sure to install the correct 
   CUDA drivers for your GPU by following this guide `Install CUDA drivers`_. 

   If you are using a cloud computing cluster with Ubuntu (e.g., on 
   `de.NBI cloud`_) and you need to setup PyTorch with NVIDIA GPU see this 
   guide :ref:`install-pytorch-with-nvidia-gpu`
   
   .. note:: 

      PyTorch is needed only if you plan to use the SpotMAX AI method for spot 
      segmentation. See the parameter :confval:`Spots segmentation method` for 
      more details.

7. **Download SpotMAX AI model weights** (optional):
   
   Download the model weights from here `3D model`_ and 
   here `2D model`_ to these paths::

        ~/spotmax_appdata/unet_checkpoints/unet2D/unet_best.pth
        ~/spotmax_appdata/unet_checkpoints/unet3D/normal_30_250_250_20_100_100/best_checkpoint.pytorch

   Alternatively, you can download them automatically with the following 
   commands::

        curl --create-dirs -o ~/spotmax_appdata/unet_checkpoints/unet2D/unet_best.pth https://hmgubox2.helmholtz-muenchen.de/index.php/s/4dxeHSLDfAbC8dA/download/unet_best.pth
        curl --create-dirs -o ~/spotmax_appdata/unet_checkpoints/unet3D/normal_30_250_250_20_100_100/best_checkpoint.pytorch https://hmgubox2.helmholtz-muenchen.de/index.php/s/eoeFcgsAMDsgTgw/download/best_checkpoint.pytorch

.. include:: _install_numba.rst

.. note:: 

  If any of the packages' installation fails, it is worth trying installing that 
  package with ``pip`` (or with ``conda`` if it fails with ``pip``). In this 
  case you will have to install the packages manually one by one. However, 
  this strategy should be used as **a very last resort**. 