.. _how-to-install:

How to install SpotMAX
======================

.. note::
    You can run spotMAX both headless or with a GUI. We recommend using ``conda`` to 
    manage virtual environments and install the latest stable spotMAX version. 
    However, if you require new features to be implemented as fast as possible 
    we recommend installing from source. 

    If you plan on contributing to the code, see the :ref:`how-to-contribute` 
    section.

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
    On macOS or Linux you can use the default Terminal.

3. **Update conda** by typing ``conda update conda``
    This will update all packages that are part of conda.

4. Create a **virtual environment** by typing ``conda create -n acdc python=3.10``
    This will create a virtual environment, which is an **isolated folder** 
    where the required libraries will be installed. 
    The virtual environment is called ``acdc`` in this case.

5. **Activate the virtual environment** by typing ``conda activate acdc``
    This will activate the environment and the terminal will know where to 
    install packages. 
    If the activation of the environment was successful, this should be 
    indicated to the left of the active path (you should see ``(acdc)`` 
    before the path).

6. **Update pip** using ``python -m pip install --upgrade pip``
    While we could use conda to install packages, spotMAX is not available 
    on conda yet, hence we will use ``pip``. 
    Pip the default package manager for Python. Here we are updating pip itself.

7. **Install spotMAX** using ``pip install "spotmax"``
    This tells pip to install spotMAX.

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

3. Clone the source code with the command ``git clone https://github.com/ElpadoCan/spotMAX.git``. 
    If you are on Windows you might need to install ``git`` first. 
    Install it from `here <https://git-scm.com/download/win>`_.

4. Navigate to the spotMAX folder with the command ``cd spotMAX``.
    The command ``cd`` stands for "change directory" and it allows you to move 
    between directories in the terminal. 
5. **Update conda** by typing ``conda update conda``
    This will update all packages that are part of conda.

6. Create a **virtual environment** by typing ``conda create -n acdc python=3.10``
    This will create a virtual environment, which is an **isolated folder** 
    where the required libraries will be installed. 
    The virtual environment is called ``acdc`` in this case.

7. Activate the virtual environment by typing ``conda activate acdc``
    This will activate the environment and the terminal will know where to 
    install packages. 
    If the activation of the environment was successful, this should be 
    indicated to the left of the active path (you should see ``(acdc)`` 
    before the path).

8. **Update pip** using ``python -m pip install --upgrade pip``
    While we could use conda to install packages, spotMAX is not available 
    on conda yet, hence we will use ``pip``. 
    Pip the default package manager for Python. Here we are updating pip itself.

9.  Install spotMAX with the command ``pip install -e .``
    The ``.`` at the end of the command means that you want to install from 
    the current folder in the terminal. This must be the ``spotMAX`` folder that you cloned before. 

Updating spotMAX installed from source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To update spotMAX installed from source, open a terminal window, navigate to the 
spotMAX folder with the command ``cd spotMAX`` and run ``git pull``.

Since you installed with the ``-e`` flag, pulling with ``git`` is enough.