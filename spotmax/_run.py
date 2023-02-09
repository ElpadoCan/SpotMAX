import sys
import os

from . import printl, GUI_INSTALLED

if GUI_INSTALLED:
    from . import gui

def run_gui(debug=False, app=None):
    from . import utils

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
        app.setWindowIcon(QIcon(":logo.svg"))
        # src_path = os.path.dirname(os.path.abspath(__file__))
        # styles_path = os.path.join(src_path, 'styles')
        # dark_orange_path = os.path.join(styles_path, '01_buttons.qss')
        # with open(dark_orange_path, mode='r') as txt:
        #     styleSheet = txt.read()
        # app.setStyleSheet(styleSheet)
        EXEC = True

    version = utils.read_version()
    win = gui.spotMAX_Win(app, debug=debug, executed=EXEC)
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

def run_cli(parser_args, debug=False):
    from . import core
    
    kernel = core.Kernel(debug=debug)
    kernel.run(
        parser_args['params'], 
        metadata_csv_path=parser_args['metadata'],
        num_numba_threads=parser_args['num_threads']
    )
    