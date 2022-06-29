import sys
import os
import argparse

from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QStyleFactory

from spotmax import gui, qrc_resources, utils, config, core

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

def run_gui(debug=False):
    print('Loading application...')
    version = utils.read_version()

    # Handle high resolution displays:
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    # Create the application
    app = QApplication(sys.argv)

    # Apply style
    app.setStyle(QStyleFactory.create('Fusion'))
    app.setWindowIcon(QIcon(":icon.svg"))
    # src_path = os.path.dirname(os.path.abspath(__file__))
    # styles_path = os.path.join(src_path, 'styles')
    # dark_orange_path = os.path.join(styles_path, '01_buttons.qss')
    # with open(dark_orange_path, mode='r') as txt:
    #     styleSheet = txt.read()
    # app.setStyleSheet(styleSheet)


    win = gui.spotMAX_Win(app, debug=debug)
    win.setVersion(version)
    win.show()

    win.logger.info('Lauching application...')
    print('\n**********************************************')
    win.logger.info(f'Welcome to spotMAX v{version}!')
    print('**********************************************\n')
    print('-----------------------------------')
    win.logger.info(
        'NOTE: If application is not visible, it is probably minimized '
        'or behind some other open window.'
    )
    print('-----------------------------------')

    sys.exit(app.exec_())

def run_cli(params_path, debug=False):
    if not os.path.exists(params_path):
        raise FileNotFoundError(
            f'The following parameters file provided does not exist: "{params_path}"'
        )
        return
    kernel = core.Kernel(debug=debug)
    kernel.init_params(params_path)

def run():
    parsers_args = cli_parser()

    params_path = parsers_args['p']
    debug = parsers_args['debug']

    if params_path:
        run_cli(params_path, debug=debug)
    else:
        run_gui(debug=debug)

if __name__ == "__main__":
    run()
