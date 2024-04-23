.. _install-from-source:

Install from source (developer version)
---------------------------------------

If you want to try out experimental features (and, if you have time, maybe report a bug or two :D), you can install the developer version from source as follows:

.. include:: _install_conda_open_terminal.rst

.. include:: _conda_create_activate_acdc.rst

7. **Clone the source code** with the following command:
   
    .. code-block:: 
    
        git clone https://github.com/ElpadoCan/spotMAX.git

    .. important::
    
        On Windows, if you get the error ``ERROR: Cannot find the command 'git'`` 
        you need to install ``git`` first. Close the terminal and install it 
        from `here <https://git-scm.com/download/win>`_. After installation, 
        you can restart from here, but **remember to activate the** ``acdc`` 
        **environment first** with the command ``conda activate acdc``.

8. **Navigate to the spotMAX folder** with the following command:
   
    .. code-block:: 
   
        cd spotMAX

    The command ``cd`` stands for "change directory" and it allows you to move 
    between directories in the terminal. 

9.  **Install spotMAX** with the following command:
   
    .. code-block:: 
   
        pip install -e "."

    The ``.`` at the end of the command means that you want to install from 
    the current folder in the terminal. This must be the ``spotMAX`` folder 
    that you cloned before. 

10. **Install the GUI libraries**:

    If you plan to use the spotMAX GUI and you never used Cell-ACDC before, 
    run the command ``acdc``. Remember to **always activate** the ``acdc`` 
    environment with the command ``conda activate acdc`` every time you 
    open a new terminal before starting Cell-ACDC.
    
    The first time you run Cell-ACDC you will be guided through the automatic 
    installation of the GUI libraries. Simply answer ``y`` in the terminal when 
    asked. 

    At the end you might have to re-start Cell-ACDC. 

.. include:: _install_numba.rst

Updating spotMAX installed from source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To update spotMAX installed from source, open a terminal window, navigate to the 
spotMAX folder with the command ``cd spotMAX`` and run ``git pull``.

Since you installed with the ``-e`` flag, pulling with ``git`` is enough.