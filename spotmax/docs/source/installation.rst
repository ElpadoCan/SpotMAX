.. _Cell-ACDC: https://github.com/SchmollerLab/Cell_ACDC

Installation instructions
=========================

You can run spotMAX both headless or with a GUI. We recommend using ``conda`` to 
manage virtual environments and install the latest stable spotMAX version. 
However, if you require new features to be implemented as fast as possible 
we recommend installing from source. 

If you plan on contributing to the code, see the `Contributing to the code`_ 
section.

Installation using Anaconda (recommended)
-----------------------------------------

1. Install `Anaconda <https://www.anaconda.com/download>`_ or `Miniconda <https://docs.conda.io/projects/miniconda/en/latest/index.html#quick-command-line-install>`_ 
    Anaconda is the standard **package manager** for Python in the scientific 
    community. It comes with a GUI for user-friendly package installation 
    and management. However, here we describe its use through the terminal. 
    Miniconda is a lightweight implementation of Anaconda without the GUI.
2. Open a **terminal**
    A terminal is a **text-based way to send instructions**. 
    On Windows, use the **Anaconda prompt**, you can find it by searching for it. 
    On macOS or Linux you can use the default Terminal.
3. **Update conda** by typing ``conda update conda``
    This will update all packages that are part of conda.
4. Create a **virtual environment** by typing ``conda create -n spotmax python=3.10``
    This will create a virtual environment, which is an **isolated folder** 
    where the required libraries will be installed. 
    The virtual environment is called ``spotmax`` in this case.
5. **Activate the virtual environment** by typing ``conda activate spotmax``
    This will activate the environment and the terminal will know where to 
    install packages. 
    If the activation of the environment was successful, this should be 
    indicated to the left of the active path (you should see ``(spotmax)`` 
    before the path).

6. **Update pip** using ``python -m pip install --upgrade pip``
    While we could use conda to install packages, spotMAX is not available 
    on conda yet, hence we will use ``pip``. 
    Pip the default package manager for Python. Here we are updating pip itself.

7. **Install spotMAX** using ``pip install "spotmax"``
    This tells pip to install spotMAX.