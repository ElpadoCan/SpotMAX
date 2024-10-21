.. _install-latest-version:

Install latest version
-----------------------

.. include:: _install_conda_open_terminal.rst

.. include:: _conda_create_activate_acdc.rst

7. **Install Cell-ACDC** latest version:

    .. code-block:: 
        
        pip install --upgrade "git+https://github.com/SchmollerLab/Cell_ACDC.git"
    
    We need to install Cell-ACDC latest version because SpotMAX heavily relies 
    on Cell-ACDC and it is very likely that it needs the latest version.

    .. important::
    
        On Windows, if you get the error ``ERROR: Cannot find the command 'git'`` 
        you need to install ``git`` first. Close the terminal and install it 
        from `here <https://git-scm.com/download/win>`_. After installation, 
        you can restart from here, but **remember to activate the** ``acdc`` 
        **environment first** with the command ``conda activate acdc``.

8.  **Install SpotMAX** from the GitHub repository with the following command:
   
    .. code-block:: 
        
        pip install "git+https://github.com/ElpadoCan/SpotMAX.git"
    
    .. tip:: 

        If you **already have the** :ref:`stable version <install-stable-version>` 
        and you want to upgrade to the latest version run the following command 
        instead:

        .. code-block::

            pip install --upgrade "git+https://github.com/ElpadoCan/SpotMAX.git"
        
    This tells pip to install SpotMAX directly from the GitHub repo.

9.  **Install the GUI libraries**:

    If you plan to use the SpotMAX GUI and you never used Cell-ACDC before, 
    run the command ``acdc``. Remember to **always activate** the ``acdc`` 
    environment with the command ``conda activate acdc`` every time you 
    open a new terminal before starting Cell-ACDC.
    
    The first time you run Cell-ACDC you will be guided through the automatic 
    installation of the GUI libraries. Simply answer ``y`` in the terminal when 
    asked. 

    At the end you might have to re-start Cell-ACDC. 

    .. include:: _gui_packages.rst

.. include:: _install_numba.rst

Updating to the latest version of SpotMAX 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To update to the latest version of SpotMAX, open the terminal, activate the 
``acdc`` environment with the command ``conda activate acdc`` and the run the 
follwing command::
        
    pip install --upgrade "git+https://github.com/ElpadoCan/SpotMAX.git"