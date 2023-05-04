import os

from . import dialogs, printl

def editFilePathsToAnalyse(formWidget):
    selectedExpPaths = dialogs.getSelectedExpPaths(
        'Select folders to analyse', parent=formWidget
    )
    if selectedExpPaths is None:
        return
    
    paths = []
    for expPath, posFoldernames in selectedExpPaths.items():
        for pos in posFoldernames:
            paths.append(os.path.join(expPath, pos))
    
    paths = '\n'.join(paths)
    formWidget.widget.setText(paths)
    formWidget.widget.setToolTip(paths)