import os

from . import html_func

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