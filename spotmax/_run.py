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
    
    from cellacdc._palettes import getPaletteColorScheme, setToolTipStyleSheet
    palette = getPaletteColorScheme(app.palette(), scheme='light')
    app.setPalette(palette)
    setToolTipStyleSheet(app, scheme='light')

    return app

def _install_acdctools():
    import subprocess
    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'install', '-U',
        'git+https://github.com/SchmollerLab/acdc_tools']
    )

def _ask_install_acdc_tools_cli():
    try:
        while True:
            answer = input(
                '>>> spotMAX detected the missing library `acdctools`. '
                'Do you want to proceed with its installation ([y]/n)? ',
            )
            if answer.lower() == 'n':
                raise ModuleNotFoundError(
                    'User aborted `acdctools` installation.'
                )
            elif answer.lower() == 'y':
                break
            else:
                print(
                    f'"{answer}" is not a valid answer. '
                    'Type "y" for yes, or "n" for no.'
                )
    except EOFError as e:
        print(
            '[ERROR]: The library `acdctools` is missing. '
            'Please install it with `pip install acdctools`'
        )

def check_cli_requirements():
    try:
        import acdctools
        return
    except Exception as e:
        pass

    _ask_install_acdc_tools_cli()
    _install_acdctools()

def _ask_install_gui_cli():
    err_msg = (
        'GUI modules are not installed. Please, install them with the '
        'command `pip install cellacdc`, or go to this link for more '
        'information: https://github.com/SchmollerLab/Cell_ACDC'
    )
    sep = '='*60
    warn_msg = (
        f'{sep}\n'
        'GUI modules are not installed. '
        'To run spotMAX GUI you need to install the package called `cellacdc`.\n'
        'To do so, run the command `pip install cellacdc`.\n\n'
        'Alternatively, you can use spotMAX in the command line interface.\n'
        'Type `spotmax -h` for help on how to do that.\n\n'
        'Do you want to install GUI modules now ([Y]/n)? '
    )
    answer = input(warn_msg)
    if answer.lower() == 'n':
        raise ModuleNotFoundError(f'{err_msg}')

def _install_gui_cli():
    import subprocess
    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'install', '-U', 'cellacdc']
    )

def _check_install_acdctools_gui(app):
    try:
        import acdctools
        return
    except Exception as e:
        pass

    if app is None:
        app = _setup_app()
    from cellacdc.myutils import _install_package_msg
    cancel = _install_package_msg('acdctools', caller_name='spotMAX')
    if cancel:
        raise ModuleNotFoundError(
            f'User aborted `acdctools` installation.'
        )
    _install_acdctools()

def _check_install_qtpy():
    try:
        import qtpy
    except ModuleNotFoundError as e:
        while True:
            txt = (
                'Since version 1.3.1 Cell-ACDC requires the package `qtpy`.\n\n'
                'You can let Cell-ACDC install it now, or you can abort '
                'and install it manually with the command `pip install qtpy`.'
            )
            print('-'*60)
            print(txt)
            answer = input('Do you want to install it now ([y]/n)? ')
            if answer.lower() == 'y' or not answer:
                import subprocess
                import sys
                subprocess.check_call(
                    [sys.executable, '-m', 'pip', 'install', '-U', 'qtpy']
                )
                break
            elif answer.lower() == 'n':
                raise e
            else:
                print(
                    f'"{answer}" is not a valid answer. '
                    'Type "y" for "yes", or "n" for "no".'
                )
    except ImportError as e:
        # Ignore that qtpy is installed but there is no PyQt bindings --> this 
        # is handled in the next block
        pass

def check_gui_requirements(app):
    from . import GUI_INSTALLED
    _check_install_qtpy()
    if GUI_INSTALLED:
        app = _check_install_acdctools_gui(app)
    else:
        _ask_install_gui_cli()
        _install_gui_cli()
        app = _check_install_acdctools_gui(app)
    return app

def run_gui(debug=False, app=None):
    check_gui_requirements(app)

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
    check_cli_requirements()

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
    