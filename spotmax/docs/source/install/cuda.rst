.. _install-pytorch-with-nvidia-gpu:

Install PyTorch with GPU support (Linux)
----------------------------------------

1. **Check NVIDIA drivers**: 
   
   First, check if the drivers are already installed by running this command::
   
    nvidia-smi

   If successful, you should see a table with information about the GPU and 
   CUDA Version. In this case, skip to point 3. 

   If the command fails to connect to the GPU, do a clean install of the drivers.

2. **Install NVIDIA drivers**:
   
   First, clean all the previously installed drivers and packages with 
   the following commands::

    sudo apt-get remove --purge '^nvidia-.*'
    sudo apt-get remove --purge '^libnvidia-.*'
    sudo apt-get remove --purge '^cuda-.*'

   Then, after it's clean, run this command::

    sudo apt-get install linux-headers-$(uname -r)

   Now you can proceed to install the NVIDIA drivers. The easiest is to use 
   the package called ``ubuntu-drivers``. Install this package with the command::

    sudo apt install ubuntu-drivers-common

   Then, list all the drivers that you can install with the following command::

    sudo ubuntu-drivers list --gpgpu

   You should see something like this::

    nvidia-driver-535-server, (kernel modules provided by nvidia-dkms-535-server)
    nvidia-driver-470, (kernel modules provided by nvidia-dkms-470)
    nvidia-driver-535, (kernel modules provided by nvidia-dkms-535)
    nvidia-driver-470-server, (kernel modules provided by nvidia-dkms-470-server)
    nvidia-driver-450-server, (kernel modules provided by nvidia-dkms-450-server)
    nvidia-driver-535-server-open, (kernel modules provided by nvidia-dkms-535-server-open)
    nvidia-driver-535-open, (kernel modules provided by nvidia-dkms-535-open) 
   
   Since we are on a server machine, we want to install the **latest server 
   version**, which in this case it is ``nvidia-driver-535-server``. To do so, 
   run this command::

    sudo ubuntu-drivers install --gpgpu nvidia:535-server
   
   After installation, you will need to **restart the machine**. This is 
   **VERY IMPORTANT**. Typically, you can do so from the web interface of the 
   cloud service that you are using. Make sure that you do a **hard reboot**. 

   After rebooting, you can check that the installation was successful, by 
   repeating point 1.

3. **Install cuDNN**:

   The cuDNN is a GPU-accelerated library for deep neural networks. We 
   recommend installing it with conda using the following command::

    conda install nvidia/label/cuda-12.1.0::cuda-nvcc
   
   .. important:: 

    Make sure that the **right environment is active**, e.g., with 
    ``conda activate acdc``. 

   After installation you can check the ``nvcc`` version with the command::

    nvcc --version

4. **Install CUDA toolkit**:
   
   .. code-block:: 

    conda install anaconda::cudatoolkit

5. **Install PyTorch**:
   
   .. code-block:: 

    conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

6. **Verify GPU support**:
   
   Run the command ``python`` and hit Enter. Then run ``import torch``, hit 
   Enter and type ``torch.cuda.is_available()``. If you see ``True`` in 
   the terminal, check that you can connect to the GPU by typing 
   ``torch.cuda.get_device_name(0)``. 
