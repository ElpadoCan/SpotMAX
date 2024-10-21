.. _run-cli:

Run SpotMAX from the command line (headless)
============================================

To run SpotMAX from the command line you will need to create a configuration 
INI file (extension ``.ini``) containing the analysis parameters. 

.. note:: 

    While it is possible to modify a INI template file or write one from scratch 
    we highly recommend to generate it using the GUI. See this section for more 
    details :ref:`how-to-run-gui`.

The configuration file is separated into sections with the same name you will 
find in the GUI. 

Once you have the configuration file, you can simply run SpotMAX in the command 
line by typing the following command::

    spotmax - p path/to/configuration_file.ini

.. important:: 

    Remember to first activate the environment where you installed SpotMAX 
    otherwise the command ``spotmax`` will not be found. 
    Refer to the installation guide for details about activating the environment 
    :ref:`how-to-install`. 

.. rubric:: Additional resources

* `Template configuration files <https://github.com/ElpadoCan/SpotMAX/tree/main/examples/ini_config_files_templates>`_ 
* :ref:`params-desc`

This is how a configuration file looks like:

.. literalinclude:: ../../../../examples/ini_config_files_templates/smFISH_yeast_tutorial_parameters.ini
    :language: ini
