import sys
import os
import argparse

from spotmax._run import run_gui, run_cli

def cli_parser():
    ap = argparse.ArgumentParser(description='spotMAX parser')

    ap.add_argument(
        '-p',
        default='',
        type=str,
        metavar='PATH_TO_PARAMS',
        help=('Path of the "_analysis_inputs.ini" or "_analysis_inputs.csv" file')
    )

    ap.add_argument(
        '-m',
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
    parsers_args = cli_parser()

    params_path = parsers_args['p']
    debug = parsers_args['debug']

    if params_path:
        run_cli(parsers_args, debug=debug)
    else:
        run_gui(debug=debug)

if __name__ == "__main__":
    run()
