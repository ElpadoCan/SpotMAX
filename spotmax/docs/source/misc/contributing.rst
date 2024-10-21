.. _GitHub page: https://github.com/ElpadoCan/SpotMAX

.. _GitHub: https://github.com/ElpadoCan/SpotMAX/issues

.. _Anaconda: https://www.anaconda.com/download

.. _Miniconda: https://docs.conda.io/projects/miniconda/en/latest/index.html#latest-miniconda-installer-links

.. _Python: https://www.python.org/downloads/

.. _Angular Conventional Commits: https://www.conventionalcommits.org

.. _how-to-contribute:

Contributing guide
==================

We welcome and we encourage contributions! 

The simplest way to contribute is to report any issues or give feedback (feature 
requests, ideas on how to improve the software, etc.) on our `GitHub`_ page. 

Another way is to contribute to the software development process. Typically, this 
is done through the process of Pull Requests (PRs) on GitHub. 

A PR is a request to include in the base repository your new code. When you write 
new code for SpotMAX, you will do this on your personal fork of the project. 
A fork is like a copy of the original code. This way you can test the new code 
on your fork and when you are happy with it you open a PR where you describe your 
contribution and you request that your code is merged with the base code. 

Setup development environment
-----------------------------

1. **Fork the base repository**:
    Go our `GitHub page`_ and click on the fork button (top-right)

2. **Clone the forked repository to your local machine**:
    
    Open a terminal and run the following command. 

    .. code-block::

        git clone https://github.com/your-username/SpotMAX
    
    Make sure to replace ``your-username`` with your GitHub username in the 
    command above. 

3. **Change directory in the terminal to the cloned folder**:

    .. code-block::

        cd SpotMAX

4. **Set the upstream remote repository to the base ``SpotMAX`` repository**:

    .. code-block::

        git remote add upstream https://github.com/ElpadoCan/SpotMAX

    This command allows you to be able to pull the latest version from the base 
    repository before pushing your changes. It is important that you work 
    on the latest version to avoid merge conflicts and for testing purposes. 
       
5. **Create a virtual development environment and activate it**:

    .. tabs::

       .. tab:: Using ``conda``
          
          Install `Anaconda`_ or `Miniconda`_ and run the following commands:

          .. code-block::

             conda create -n acdc-dev python=3.10
             conda activate acdc-dev

       .. tab:: Using ``venv``
         
          Install `Python`_ and run the following commands:

          .. code-block::

             python -m venv <path-to-env>
             source <path-to-env>/bin/activate

6. **Install the forked SpotMAX in developer mode**:
    Make sure to be in the ``SpotMAX`` folder in the terminal before running the 
    following command:

    .. code-block::

        pip install -e "."
    
    .. note::

        If you are planning to contribute to the GUI, make sure to run SpotMAX 
        at least once and let it install the required GUI libraries. 

Develop your contribution
-------------------------

To develop your contribution you first need to create a branch on your forked 
repository. Then you will push the changes to the branch and create a Pull Request 
(PR) on GitHub. 

A member of our team will review your PR, propose changes if needed, and once 
everything looks good your PR will be merged on the main branch of the base 
repository. 

These are the steps:

1. **Update your cloned fork to the latest version**:
    Open a terminal and run the following commands:

    .. code-block:: 

        cd SpotMAX
        git checkout main
        git pull upstream main

2. **Create a branch with the name of the contribution**:

    .. code-block:: 

        git checkout -b contribution-name

3. **Commit your changes locally to the forked cloned repository**:

    .. code-block:: 

        git add .
        git commit -m "commit message"
    
    .. important::

        When writing the commit message, please follow the 
        `Angular Conventional Commits`_ specification.

4. **Open a Pull Request**:
    To open a Pull Request go to the GitHub page of your forked repository and 
    you will see a green button on the top-left to open the PR. Click that 
    button and add a description about your contribution. 

    .. tip::

        To modify the PR you can simply commit and push to the same branch. GitHub 
        will automatically update the PR. 