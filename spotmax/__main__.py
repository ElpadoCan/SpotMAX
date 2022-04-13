import sys

from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QStyleFactory

from spotmax import gui, qrc_resources

def run():
    print('Loading application...')
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


    win = gui.spotMAX_Win(app)
    win.show()

    # Run the event loop
    win.logger.info('Lauching application...')
    win.logger.info(
        'Done. If application GUI is not visible, it is probably minimized, '
        'behind some other open window, or on second screen.'
    )
    sys.exit(app.exec_())

if __name__ == "__main__":
    run()
