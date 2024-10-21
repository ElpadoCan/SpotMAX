.. _install-stable-version:

Install stable version
----------------------

.. include:: _install_conda_open_terminal.rst

.. include:: _conda_create_activate_acdc.rst

7.  **Install SpotMAX** with the following command:
   
    .. code-block:: 
        
        pip install "spotmax"
        
    This tells pip to install SpotMAX.

8.  **Install the GUI libraries**:

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

Updating to the latest stable version of SpotMAX 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To update to the latest version of SpotMAX, open the terminal, activate the 
``acdc`` environment with the command ``conda activate acdc`` and the run the 
follwing command::
        
    pip install --upgrade spotmax