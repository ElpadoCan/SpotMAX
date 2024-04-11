.. _install-latest-version:

Install latest version
-----------------------

1. Install `Anaconda <https://www.anaconda.com/download>`_ or `Miniconda <https://docs.conda.io/projects/miniconda/en/latest/index.html#latest-miniconda-installer-links>`_ 
    Anaconda is the standard **package manager** for Python in the scientific 
    community. It comes with a GUI for user-friendly package installation 
    and management. However, here we describe its use through the terminal. 
    Miniconda is a lightweight implementation of Anaconda without the GUI.

2. Open a **terminal**
    Roughly speaking, a terminal is a **text-based way to run instructions**. 
    On Windows, use the **Anaconda prompt**, you can find it by searching for it. 
    On macOS or Linux you can use the default Terminal app.

3. **Update conda** by running the following command:
    
    .. code-block:: 
    
        conda update conda
    
    This will update all packages that are part of conda.

4. **Create a virtual environment** with the following command:
   
    .. code-block:: 
   
        conda create -n acdc python=3.10

    This will create a virtual environment, which is an **isolated folder** 
    where the required libraries will be installed. 
    The virtual environment is called ``acdc`` in this case.

5. **Activate the virtual environment** with the following command:
   
    .. code-block:: 
   
        conda activate acdc
    
    This will activate the environment and the terminal will know where to 
    install packages. 
    If the activation of the environment was successful, this should be 
    indicated to the left of the active path (you should see ``(acdc)`` 
    before the path).

    .. important:: 

       Before moving to the next steps make sure that you always activate 
       the ``acdc`` environment. If you close the terminal and reopen it, 
       always run the command ``conda activate acdc`` before installing any 
       package. To know whether the right environment is active, the line 
       on the terminal where you type commands should start with the text 
       ``(acdc)``, like in this screenshot:

       .. tabs::

            .. tab:: Windows

                .. figure:: ../images/conda_activate_acdc_windows.png
                    :width: 100%

                    Anaconda Prompt after activating the ``acdc`` environment 
                    with the command ``conda activate acdc``.
            
            .. tab:: macOS

                .. figure:: ../images/conda_activate_acdc_macOS.png
                    :width: 100%

                    Terminal app after activating the ``acdc`` environment 
                    with the command ``conda activate acdc``.
            


6. **Update pip** with the following command:
   
    .. code-block:: 
   
        python -m pip install --upgrade pip
    
    While we could use conda to install packages, spotMAX is not available 
    on conda yet, hence we will use ``pip``. 
    Pip the default package manager for Python. Here we are updating pip itself.

7. **Install Cell-ACDC** latest version:

    .. code-block:: 
        
        pip install --upgrade "git+https://github.com/SchmollerLab/Cell_ACDC.git"
    
    We need to install Cell-ACDC latest version because spotMAX heavily relies 
    on Cell-ACDC and it is very likely that it needs the latest version.

    .. important::
    
        On Windows, if you get the error ``ERROR: Cannot find the command 'git'`` 
        you need to install ``git`` first. Close the terminal and install it 
        from `here <https://git-scm.com/download/win>`_. After installation, 
        you can restart from here, but **remember to activate the** ``acdc`` 
        **environment first** with the command ``conda activate acdc``.

8.  **Install spotMAX** from the GitHub repository with the following command:
   
    .. code-block:: 
        
        pip install "git+https://github.com/ElpadoCan/spotMAX.git"
    
    .. tip:: 

        If you **already have the stable version** and you want to upgrade to the 
        latest version run the following command instead:

        .. code-block::

            pip install --upgrade "git+https://github.com/ElpadoCan/spotMAX.git"
        
    This tells pip to install spotMAX directly from the GitHub repo.

9.  **Install the GUI libraries**:

    If you plan to use the spotMAX GUI and you never used Cell-ACDC before, 
    run the command ``acdc``. Remember to **always activate** the ``acdc`` 
    environment with the command ``conda activate acdc`` every time you 
    open a new terminal before starting Cell-ACDC.
    
    The first time you run Cell-ACDC you will be guided through the automatic 
    installation of the GUI libraries. Simply answer ``y`` in the terminal when 
    asked. 

    At the end you might have to re-start Cell-ACDC. 

.. include:: _install_numba.rst

Updating to the latest version of spotMAX 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To update to the latest version of spotMAX, open the terminal, activate the 
``acdc`` environment with the command ``conda activate acdc`` and the run the 
follwing command::
        
    pip install --upgrade "git+https://github.com/ElpadoCan/spotMAX.git"