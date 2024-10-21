Optional: install Numba
~~~~~~~~~~~~~~~~~~~~~~~

SpotMAX can take advantage of multiple CPU cores using the package ``numba``. 
For more details, see the `Numba documentation <https://numba.readthedocs.io/en/stable/index.html>`_. 

This typically increases execution speed. **After installing SpotMAX**, consider 
installing ``numba`` with the following command:

.. tabs::

    .. tab:: conda
        
        .. code-block::
        
            conda install numba
    
    .. tab:: pip
        
        .. code-block::
    
            pip install numba

.. note::

    You can set the number of CPU cores used by ``numba`` in the INI 
    configuration file using the :confval:`Number of threads used by numba` 
    parameter as follows:

    .. code-block::ini

        [Configuration]
        Number of threads used by numba = 4


