.. _example notebooks: https://github.com/ElpadoCan/SpotMAX/tree/main/examples/notebooks
.. _GitHub page: https://github.com/ElpadoCan/SpotMAX/issues


.. _how-to-run-py-pkg:

Run SpotMAX as a Python package
===============================

You can run SpotMAX as a Python package by using the **high-level functions** 
available in the ``pipe`` module. 

See this section for more details about the available functions :ref:`api`. 

Feel free to explore also the functions from the other modules, however, 
documentation for those modules is still work in progress :(. If you need help 
with those, submit an issue on our `GitHub page`_ specifying which 
functions you would like to use.

As a regular Python package you can also use SpotMAX in a **Jupyter notebook**. 

You can find `example notebooks`_ on our GitHub page.

Generate a Python script from an INI file
-----------------------------------------

To create a Python entry-point script for an existing analysis configuration::

    from spotmax import config

    script_path = config.create_python_script_from_ini(
        "path/to/analysis_parameters.ini"
    )

The generated script calls ``pipe.Model.predict`` for each image frame, then
runs the remaining INI-configured batch analysis and saves the usual SpotMAX
outputs. It writes a working INI and per-position predicted-coordinate CSVs so
the run can be reproduced.
