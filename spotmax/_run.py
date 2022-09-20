import sys
import os

from . import core, utils, gui


def run_gui(debug=False, app=None):
    EXEC = False
    if app is None:
        print('Loading application...')
        from PyQt5.QtGui import QIcon
        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import QApplication, QStyleFactory
    
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
        EXEC = True

    version = utils.read_version()
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

    if EXEC:
        sys.exit(app.exec_())
    else:
        return win

def run_cli(parsers_args, debug=False):
    params_path = parsers_args['p']
    metadata_csv_path = parsers_args['m']
    if not os.path.exists(params_path):
        raise FileNotFoundError(
            f'The following parameters file provided does not exist: "{params_path}"'
        )
        return
    kernel = core.Kernel(debug=debug)
    kernel.init_params(params_path, metadata_csv_path=metadata_csv_path)