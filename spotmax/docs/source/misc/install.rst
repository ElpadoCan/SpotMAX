.. _Cell-ACDC: https://cell-acdc.readthedocs.io/en/latest/index.html

.. note::

    You can run spotMAX both headless or with the GUI. We recommend 
    using ``conda`` to manage virtual environments and install the latest 
    stable spotMAX version. However, if you require new features to be 
    implemented as fast as possible we recommend installing from source. 

    If you plan to contribute to the code, see the :ref:`how-to-contribute` 
    section.

.. _how-to-install:

How to install SpotMAX
======================

Here you will find a detailed guide on how to install spotMAX. In general, 
you should be fine with installing the stable version, however, spotMAX 
development is still quite rapid and if you want to try out the latest 
features we recommend installing the latest version. On the other hand, 
installing from source is required only if you plan to contribute to spotMAX
development. In that case see our :ref:`how-to-contribute`.

.. tip:: 
    
    If you are **new to Python** or you need a **refresher** on how to manage 
    scientific Python environments, I highly recommend reading 
    `this guide <python-guide>`__ by Dr. Robert Haase.

* `Install stable version <install-stable-version>`_
* `Install latest version <install-latest-version>`_
* `Install from source <install-from-source>`_

.. _install-stable-version:

Install stable version
----------------------

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


6. **Update pip** with the following command:
   
    .. code-block:: 
   
        python -m pip install --upgrade pip
    
    While we could use conda to install packages, spotMAX is not available 
    on conda yet, hence we will use ``pip``. 
    Pip the default package manager for Python. Here we are updating pip itself.

7.  **Install spotMAX** with the following command:
   
    .. code-block:: 
        
        pip install "spotmax"
        
    This tells pip to install spotMAX.

Updating to the latest stable version of spotMAX 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To update to the latest version of spotMAX, open the terminal, activate the 
``acdc`` environment with the command ``conda activate acdc`` and the run the 
follwing command::
        
    pip install --upgrade spotmax


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

8.  **Install spotMAX** from the GitHub repository with the following command:
   
    .. code-block:: 
        
        pip install "git+https://github.com/ElpadoCan/spotMAX.git"
        
    This tells pip to install spotMAX directly from the GitHub repo.

Updating to the latest version of spotMAX 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To update to the latest version of spotMAX, open the terminal, activate the 
``acdc`` environment with the command ``conda activate acdc`` and the run the 
follwing command::
        
    pip install --upgrade "git+https://github.com/ElpadoCan/spotMAX.git"


.. _install-from-source:

Install from source (developer version)
---------------------------------------

If you want to try out experimental features (and, if you have time, maybe report a bug or two :D), you can install the developer version from source as follows:

1. Install `Anaconda <https://www.anaconda.com/download>`_ or `Miniconda <https://docs.conda.io/projects/miniconda/en/latest/index.html#latest-miniconda-installer-links>`_ 
    Anaconda is the standard **package manager** for Python in the scientific 
    community. It comes with a GUI for user-friendly package installation 
    and management. However, here we describe its use through the terminal. 
    Miniconda is a lightweight implementation of Anaconda without the GUI.

2. Open a **terminal**
    Roughly speaking, a terminal is a **text-based way to run instructions**. 
    On Windows, use the **Anaconda prompt**, you can find it by searching for it. 
    On macOS or Linux you can use the default Terminal.

3. **Clone the source code** with the following command:
   
    .. code-block:: 
    
        git clone https://github.com/ElpadoCan/spotMAX.git

    If you are on Windows you might need to install ``git`` first. 
    Install it from `here <https://git-scm.com/download/win>`_.

4. **Navigate to the spotMAX folder** with the following command:
   
    .. code-block:: 
   
        cd spotMAX

    The command ``cd`` stands for "change directory" and it allows you to move 
    between directories in the terminal. 

5. **Update conda** with the following command:
   
    .. code-block:: 

        conda update conda
    
    This will update all packages that are part of conda.

6. Create a **virtual environment** with the following command:
   
    .. code-block:: 
    
        conda create -n acdc python=3.10

    This will create a virtual environment, which is an **isolated folder** 
    where the required libraries will be installed. 
    The virtual environment is called ``acdc`` in this case.

7. **Activate the virtual environment** with the following command:
   
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

8. **Update pip** with the following command:
   
    .. code-block:: 
   
        python -m pip install --upgrade pip
    
    While we could use conda to install packages, spotMAX is not available 
    on conda yet, hence we will use ``pip``. 
    Pip the default package manager for Python. Here we are updating pip itself.

9.  **Install spotMAX** with the following command:
   
    .. code-block:: 
   
        pip install -e "."

    The ``.`` at the end of the command means that you want to install from 
    the current folder in the terminal. This must be the ``spotMAX`` folder 
    that you cloned before. 


Updating spotMAX installed from source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To update spotMAX installed from source, open a terminal window, navigate to the 
spotMAX folder with the command ``cd spotMAX`` and run ``git pull``.

Since you installed with the ``-e`` flag, pulling with ``git`` is enough.