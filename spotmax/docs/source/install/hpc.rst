.. _Cell-ACDC: https://cell-acdc.readthedocs.io/en/latest/index.html
.. _envs_gh_url: https://github.com/ElpadoCan/spotMAX/tree/main/tests
.. _pyenv: https://github.com/pyenv/pyenv
.. _miniconda: https://docs.anaconda.com/free/miniconda/#quick-command-line-install
.. _pytorch: https://pytorch.org/get-started/locally/

.. _install-on-hpc:

Install on HPC cluster
----------------------

HPC cluster often do not have a desktop environment, meaning that you need to 
install the headless version of spotMAX. 

Since most of the HPC clusters run on some Linux-based OS, we recommend using 
``conda`` not only to manage the environments, but also to install the 
dependencies. 

That means installing all the dependencies **first** and then install `Cell-ACDC`_ 
and spotMAX **without dependencies**. 

In `this <envs_gh_url>`_ folder, you will find the following files:

* ``conda_env_headless.yml`` to install dependencies with ``conda``.
* ``requirements_headless.txt`` to install dependencies with ``pip``. Note that 
  ``pip`` can also be used within a ``conda`` environment.

Follow these steps to install spotMAX headless:

1. **Copy environment file(s)**
   
   Copy the files above on a folder on the cluster or download them automatically 
   from the terminal with the following command::

    curl -O 

2. **Install the package manager**
   
   .. tabs:: 

        .. tab:: conda

            If ``conda`` is not already installed on the cluster, install 
            Miniconda by following `this guide <miniconda>`_.
        
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
   
   Some of spotMAX dependencies are available only from ``pip`` which means 
   you need to install them manually with the following commands::

    pip install --no-deps "git+https://github.com/SchmollerLab/Cell_ACDC.git"
    pip install "git+https://github.com/ElpadoCan/pytorch3dunet.git"

5. **Install PyTorch** (optional):

    To install PyTorch follow `this guide <pytorch>`_.
   
   .. note:: 

      PyTorch is needed only if you plan to use the spotMAX AI method for spot 
      segmentation. See the parameter :confval:`Spots segmentation method` for 
      more details.



.. note:: 

  If any of the packages' installation fails, it is worth trying installing that 
  package with ``pip`` (or with ``conda`` if it fails with ``pip``). In this 
  case you will have to install the packages manually one by one. However, 
  this strategy should be used as **a very last resort**. 