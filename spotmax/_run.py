import sys
import os

from . import printl, spotmax_path, resources_folderpath

def run_gui(debug=False, app=None, mainWin=None, launcherSlot=None):
    from cellacdc._run import _setup_gui_libraries, _setup_app
    
    _setup_gui_libraries(caller_name='SpotMAX')

    import spotmax
    spotmax.is_cli = False
    
    from . import read_version
    from . import gui
    from qtpy import QtCore

    EXEC = False
    if app is None:
        from spotmax import icon_path, logo_path
        app, splashScreen = _setup_app(
            icon_path=icon_path, logo_path=logo_path, splashscreen=True
        )
        EXEC = True
    
    version = read_version()
    win = gui.spotMAX_Win(
        app, debug=debug, executed=EXEC, version=version, mainWin=mainWin,
        launcherSlot=launcherSlot
    )
    win.run()
    win.logger.info(f'Using Qt version {QtCore.__version__}')

    win.logger.info('Lauching application...')
    welcome_text = (
        '**********************************************\n'
        f'Welcome to SpotMAX v{version}!\n'
        '**********************************************\n'
        '----------------------------------------------\n'
        'NOTE: If application is not visible, it is probably minimized '
        'or behind some other open window.\n'
        '-----------------------------------'
    )
    win.logger.info(welcome_text)

    try:
        splashScreen.close()
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
        report_filepath=report_filepath,
        disable_final_report=parser_args['disable_final_report'],
        num_numba_threads=parser_args['num_threads'],
        force_default_values=parser_args['force_default_values'],
        force_close_on_critical=parser_args['raise_on_critical'],
        parser_args=parser_args
    )
    