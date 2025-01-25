import os

from . import GUI_INSTALLED

if GUI_INSTALLED:
    from cellacdc import widgets as acdc_widgets
    from . import html_func

def informationSpotmaxAnalysisStart(ini_filepath, qparent=None):
    ini_filepath = ini_filepath.replace('\\', os.sep)
    ini_filepath = ini_filepath.replace('/', os.sep)
    txt = html_func.paragraph(f"""
        SpotMAX analysis will now <b>run in the terminal</b>. All progress 
        will be displayed there.<br><br>
        Make sure to <b>keep an eye on the terminal</b> since it might require 
        your attention.<br><br>
        
        NOTE: If you prefer to run this analysis manually in any terminal of 
        your choice run the following command:<br>
    """)
    msg = acdc_widgets.myMessageBox(wrapText=False)
    msg.information(
        qparent, 'Analysis will run in the terminal', txt,
        buttonsTexts=('Cancel', 'Ok, run now!'),
        commands=(f'spotmax -p "{ini_filepath}"',)
    )
    return msg.cancel, ini_filepath

def informationComputeFeaturesFinished(edited_df_filename, qparent=None):
    txt = html_func.paragraph(fr"""
        Computing features of edited results finished!<br><br>
        The new tables have been saved in each edited Position folder<br>
        in the <code>Position_n/spotMAX_output folder</code> with filename 
        <code>edited_df_filename</code>.
    """)
    msg = acdc_widgets.myMessageBox(wrapText=False)
    msg.information(qparent, 'Computing features finished', txt)

def warnNoneOfLoadedPosResultsEdited(qparent=None):
    txt = html_func.paragraph("""
        None of the loaded Positions have edited results.<br><br>
        Computing features process cancelled.
    """)
    msg = acdc_widgets.myMessageBox(wrapText=False)
    msg.warning(qparent, 'No need to compute features', txt)

def askUseSavedRefChMask(refChSegmEndName, qparent=None):
    txt = html_func.paragraph("""
        SpotMAX detected that the reference channel masks were saved as part of 
        the loaded analysis.<br><br>
        When computing the spots features of the edited redults, SpotMAX can 
        <b>load the saved masks</b><br>
        instead of segmenting the reference channel again 
        (hence <b>saving computation time</b>).<br><br>
        What do you want to do? 
    """)
    buttonsTexts = (
        'Segment reference channel again', 
        'Use saved reference channel masks'
    )
    msg = acdc_widgets.myMessageBox(wrapText=False)
    _, useSavedRefChSegmButton  = msg.question(
        qparent, 'Load saved ref. channel masks?', txt, 
        buttonsTexts=buttonsTexts
    )
    return msg.clickedButton == useSavedRefChSegmButton