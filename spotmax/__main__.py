import sys
import os
import argparse

from spotmax._run import run_gui, run_cli

def cli_parser():
    ap = argparse.ArgumentParser(description='spotMAX parser')

    ap.add_argument(
        '-c', '--cli',
        action='store_true',
        help=('Flag to run spotMAX in the command line')
    )

    ap.add_argument(
        '-p', '--params',
        default='',
        type=str,
        metavar='PATH_TO_PARAMS',
        help=('Path of the "_analysis_inputs.ini" or "_analysis_inputs.csv" file')
    )

    ap.add_argument(
        '-m', '--metadata',
        default='',
        type=str,
        metavar='PATH_TO_METADATA_CSV',
        help=('Path of the "_metadata.csv" file')
    )

    ap.add_argument(
        '-d', '--debug',
        action='store_true',
        help=(
            'Used for debugging. Test code with '
            '"if self.debug: <debug code here>"'
        )
    )

    # Add dummy argument for stupid Jupyter
    ap.add_argument(
        '-f', help=('Dummy argument required by notebooks. Do not use.')
    )
    return vars(ap.parse_args())

def run():
    parser_args = cli_parser()

    PARAMS_PATH = parser_args['params']
    DEBUG = parser_args['debug']
    RUN_CLI = parser_args['cli']

    if RUN_CLI and not PARAMS_PATH:
        raise FileNotFoundError(
            '[ERROR]: To run spotMAX from the command line you need to provide a path to the "_analysis_inputs.ini" or "_analysis_inputs.csv" file')

    if PARAMS_PATH or RUN_CLI:
        run_cli(parser_args, debug=DEBUG)
    else:
        run_gui(debug=DEBUG)

if __name__ == "__main__":
    run()
