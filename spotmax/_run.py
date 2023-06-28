import sys
import os

from . import printl, spotmax_path

def _setup_app():
    print('Loading application...')
    from qtpy import QtGui, QtWidgets, QtCore

    class SpotMaxSPlashScreen(QtWidgets.QSplashScreen):
        def __init__(self):
            super().__init__()
            resources_path = os.path.join(spotmax_path, 'resources')
            logo_path = os.path.join(resources_path, 'spotMAX_logo.png')
            self.setPixmap(QtGui.QPixmap(logo_path))

    # Handle high resolution displays:
    if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    # Needed by pyqtgraph with display resolution scaling
    try:
        QtWidgets.QApplication.setAttribute(
            QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    except Exception as e:
        pass
    
    # Create the application
    app = QtWidgets.QApplication([])
    app.setStyle(QtWidgets.QStyleFactory.create('Fusion'))
    hsl_window = app.palette().color(QtGui.QPalette.Window).getHsl()
    is_OS_dark_mode = hsl_window[2] < 100
    app.setPalette(app.style().standardPalette())

    app.setWindowIcon(QtGui.QIcon(":icon_spotmax.ico"))

    # Launch splashscreen
    splashScreen = SpotMaxSPlashScreen()
    splashScreen.setWindowIcon(QtGui.QIcon(":icon_spotmax.ico"))
    splashScreen.setWindowFlags(
        QtCore.Qt.WindowStaysOnTopHint 
        | QtCore.Qt.SplashScreen 
        | QtCore.Qt.FramelessWindowHint
    )
    splashScreen.show()
    splashScreen.raise_()

    app._splashScreen = splashScreen
    
    from cellacdc import settings_csv_path
    import pandas as pd
    df_settings = pd.read_csv(settings_csv_path, index_col='setting')
    isUserColorScheme = 'colorScheme' in df_settings.index
    if isUserColorScheme:
        scheme = df_settings.at['colorScheme', 'value']
    elif is_OS_dark_mode:
        scheme = 'dark'
    else:
        scheme = 'light'
    from cellacdc._palettes import getPaletteColorScheme, setToolTipStyleSheet
    palette = getPaletteColorScheme(app.palette(), scheme=scheme)
    app.setPalette(palette)
    setToolTipStyleSheet(app, scheme=scheme)

    return app

def run_gui(debug=False, app=None):
    from cellacdc._run import _setup_gui
    
    _setup_gui()

    from . import read_version
    from . import gui

    EXEC = False
    if app is None:
        app = _setup_app()
        EXEC = True

    version = read_version()
    win = gui.spotMAX_Win(app, debug=debug, executed=EXEC, version=version)
    win.run()

    win.logger.info('Lauching application...')
    welcome_text = (
        '**********************************************\n'
        f'Welcome to spotMAX v{version}!\n'
        '**********************************************\n'
        '----------------------------------------------\n'
        'NOTE: If application is not visible, it is probably minimized '
        'or behind some other open window.\n'
        '-----------------------------------'
    )
    win.logger.info(welcome_text)

    try:
        app._splashScreen.close()
    except Exception as e:
        pass

    if EXEC:
        sys.exit(app.exec_())
    else:
        return win

def run_cli(parser_args, debug=False):
    from . import core
    
    kernel = core.Kernel(debug=debug)
    parser_args = kernel.check_parsed_arguments(parser_args)

    report_filepath = os.path.join(
        parser_args['report_folderpath'], parser_args['report_filename']
    )
    kernel.run(
        parser_args['params'], 
        metadata_csv_path=parser_args['metadata'],
        report_filepath=report_filepath,
        disable_final_report=parser_args['disable_final_report'],
        num_numba_threads=parser_args['num_threads'],
        force_default_values=parser_args['force_default_values'],
        force_close_on_critical=parser_args['raise_on_critical'],
        parser_args=parser_args
    )
    