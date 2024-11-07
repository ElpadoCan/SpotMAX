import os

from . import html_func, utils

def warn_background_value_is_zero(logger_func, logger_warning_report=None):
    text = (
        'Background value is 0 --> '
        'spot center intensity to background ratio is infinite'
    )
    print('')
    logger_func(f'[WARNING]: {text}')
    
    if logger_warning_report is None:
        return
    
    logger_warning_report(text)

def warnSpotmaxOutFolderDoesNotExist(spotmax_out_path, qparent=None):
    from cellacdc import widgets
    
    txt = html_func.paragraph(f"""
        The <code>spotMAX_output</code> folder below <b>does not 
        exist</b>.<br><br>
        SpotMAX results cannot be loaded.
    """)
    
    msg = widgets.myMessageBox(wrapText=False)
    msg.warning(
        qparent, 'SpotMAX folder not found', txt, 
        commands=(spotmax_out_path,),
        path_to_browse=os.path.dirname(spotmax_out_path)
    )

def warnSpotsDetectedOutsideCells(segm_endname, qparent=None):
    from cellacdc import widgets
    
    txt = html_func.paragraph(f"""
        WARNING: Some spots were detected <b>outside of the segmented 
        objects</b> (see segm. file `{segm_endname}`).<br><br>
        To make sure these spots are detected deactivate the 
        following parameter:<br><br>
        <code>Skip objects where segmentation failed = False</code> 
    """)
    
    msg = widgets.myMessageBox(wrapText=False)
    msg.warning(qparent, 'Spots detected outside objects', txt)

def warnNeuralNetNotInitialized(qparent=None, model_type='SpotMAX AI'):
    from cellacdc import widgets
    
    txt = html_func.paragraph(f"""
        {model_type} <b>parameters</b> were <b>not initialized</b>.<br><br>
        
        You need to <b>initialize the model's parameters</b> by clicking on the settings 
        button on the right of the selection box<br>
        at the <code>Spots segmentation method</code> parameter.
    """)
    
    msg = widgets.myMessageBox(wrapText=False)
    msg.warning(qparent, 'Model parameters not initialized', txt)

def log_files_in_folder(folderpath: os.PathLike, logger_func=print):
    files = utils.listdir(folderpath)
    files_str = '\n'.join([f'* {file}' for file in files])
    head_sep = '-'*100
    foot_sep = '='*100
    logger_func(
        f'{head_sep}\n'
        f'Files found in the folder "{folderpath}":\n\n'
        f'{files_str}\n'
        f'{foot_sep}\n'
    )