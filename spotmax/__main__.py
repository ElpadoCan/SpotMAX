import sys
import os
import argparse

import site
sitepackages = site.getsitepackages()
site_packages = [p for p in sitepackages if p.endswith('site-packages')][0]

spotmax_path = os.path.dirname(os.path.abspath(__file__))
spotmax_installation_path = os.path.dirname(spotmax_path)

if spotmax_installation_path != site_packages:
    # Running developer version. Delete spotmax folder from site_packages 
    # if present from a previous installation of spotmax from PyPi
    spotmax_path_pypi = os.path.join(site_packages, 'spotmax')
    if os.path.exists(spotmax_path_pypi):
        import shutil
        try:
            shutil.rmtree(spotmax_path_pypi)
        except Exception as err:
            print(err)
            print(
                '[ERROR]: Previous SpotMAX installation detected. '
                f'Please, manually delete this folder and re-start the software '
                f'"{spotmax_path_pypi}". '
                'Thank you for you patience!'
            )
            exit()
        print('*'*60)
        input(
            '[WARNING]: SpotMAX had to clean-up and older installation. '
            'Please, re-start the software. Thank you for your patience! '
            '(Press any key to exit). '
        )
        exit()


from spotmax._run import run_gui, run_cli
from spotmax import help_text, GUI_INSTALLED

def cli_parser():
    ap = argparse.ArgumentParser(
        prog='SpotMAX', description=help_text, 
        formatter_class=argparse.RawTextHelpFormatter
    )

    ap.add_argument(
        '-p', '--params',
        default='',
        type=str,
        metavar='PATH_TO_PARAMS',
        help=('Path of the ".ini" or "_analysis_inputs.csv" file')
    )
    
    ap.add_argument(
        '-v', '--version',
        action='store_true',
        help=('Version and installation location')
    )
    
    ap.add_argument(
        '-l', '--log_filepath',
        default='',
        type=str,
        metavar='LOG_FILEPATH',
        help=('Path of an additional log file')
    )

    # NOTE: the user doesn't need to pass `-c`` because passing the path to the 
    # params is enough. However, passing `-c`` without path to params will 
    # raise an error with the explanation that the parameters file is 
    # mandatory in command line.
    ap.add_argument(
        '-c', '--cli',
        action='store_true',
        help=(
            'Flag to run SpotMAX in the command line.'
            'Not required if you pass the `--params` argument.'
        )
    )

    ap.add_argument(
        '-d', '--debug',
        action='store_true',
        help=(
            'Used for debugging. Test code with '
            '"if self.debug: <debug code here>"'
        )
    )

    return vars(ap.parse_args())

def run():
    # print('Setting up required libraries...')
    from spotmax import error_up_str
    from cellacdc._run import _install_tables
    requires_restart = _install_tables(parent_software='SpotMAX')
    if requires_restart:
        exit(
            '[NOTE]: SpotMAX had to install a required library and needs to be '
            'restarted. Thank you for you patience!. '
        )

    parser_args = cli_parser()

    PARAMS_PATH = parser_args['params']
    DEBUG = parser_args['debug']
    RUN_CLI = parser_args['cli']
    DISPLAY_VERSION = parser_args['version']
    
    if DISPLAY_VERSION:
        from cellacdc.myutils import get_info_version_text as acdc_info
        from spotmax.utils import get_info_version_text
        acdc_info_txt = acdc_info()
        
        info_txt = get_info_version_text(include_platform=False)
        
        print(acdc_info_txt)
        print(info_txt)
        return

    if RUN_CLI and not PARAMS_PATH:
        error_msg = (
            '[ERROR]: To run SpotMAX from the command line you need to '
            'provide a path to the "_analysis_inputs.ini" or '
            '"_analysis_inputs.csv" file. To run the GUI use the command '
            f'`spotmax -g`{error_up_str}'
        )
        raise FileNotFoundError(error_msg)

    if PARAMS_PATH:
        run_cli(parser_args, debug=DEBUG)
    else:
        run_gui(debug=DEBUG)

if __name__ == "__main__":
    run()
