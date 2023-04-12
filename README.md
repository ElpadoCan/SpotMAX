# <a href="https://github.com/ElpadoCan/spotMAX_v2/blob/main/spotmax/resources/logo.svg"><img src="https://raw.githubusercontent.com/ElpadoCan/spotMAX_v2/main/spotmax/resources/logo.svg?token=GHSAT0AAAAAACBKBSCCI6NWGGFS36CZIOUCZBWUNIQ" width="80" height="80"></a> spotMAX

### A Python package for automatic **detection**, and **quantification** of fluorescent spot in microscopy data

*Written in Python 3 by [Francesco Padovani](https://github.com/ElpadoCan)


## Install from source

If you want to try out experimental features (and, if you have time, maybe report a bug or two :D), you can install the developer version from source as follows:

1. Install [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
2. Open a terminal and navigate to a folder where you want to download Cell-ACDC. If you are on Windows you need to use the "Anaconda Prompt" as a terminal. You should find it by searching for "Anaconda Prompt" in the Start menu.
3. Clone the source code with the command `git clone https://github.com/ElpadoCan/spotMAX.git`. If you are on Windows you might need to install `git` first. Install it from [here](https://git-scm.com/download/win).
4. Navigate to the `spotMAX` folder with the command `cd spotMAX`.
5. Update conda with `conda update conda`. Optionally, consider removing unused packages with the command `conda clean --all`
6. Create a new conda environment with the command `conda create -n acdc python=3.9`
7. Activate the environment with the command `conda activate acdc`
8. Upgrade pip with the command `python -m pip install --upgrade pip`
9. Install spotMAX with the command `pip install -e .`. The `.` at the end of the command means that you want to install from the current folder in the terminal. This must be the `spotMAX` folder that you cloned before. 

### Updating spotMAX installed from source

To update spotMAX installed from source, open a terminal window, navigate to the spotMAX folder and run the command
```
git pull
```
Since you installed with the `-e` flag, pulling with `git` is enough.

## Run spotMAX from the command-line iterface

To run spotMAX from the command-line, you need to create the parameters file. See [here]() an example file.

Place the .ini file in a folder, activate the `acdc` environment and thne run the command `spotmax -p <path_to_ini_file>`.