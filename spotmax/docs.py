from . import config, html_func

def get_href_tags():
    params = config.analysisInputsParams()
    href_tags = {}
    for section, anchors in params.items():
        for anchor, options in anchors.items():
            desc = options['desc']
            tag_info = f'a href="{section};{anchor}"'
            href_tags[anchor] = html_func.tag(desc, tag_info)
    return href_tags

def paramsInfoText():
    paramsInfoText = {
    'spotsFilePath': (
        'Path of the image file with the <b>spots channel signal</b>.<br><br>'
        'Allowed <b>file formats</b>: .npy, .npz, .h5, .png, .tif, .tiff, '
        '.jpg, .jpeg, .mov, .avi, and .mp4'
    ),
    'segmFilePath': (
        '<b>OPTIONAL</b>: Path of the file with the <b>segmentation masks of '
        'the objects of interest</b>. Typically the objects are the <b>cells '
        'or the nuclei</b>, but it can be any object.<br><br>'
        'While this is optional, <b>it improves accuracy</b>, because spotMAX will '
        'detect only the spots that are inside the segmented objects.<br><br>'
        'It needs to be a 2D or 3D (z-stack) array, eventually with '
        'an additional dimension for frames over time.<br>'
        'The Y and X dimensions <b>MUST be the same</b> as the spots '
        'or reference channel images.<br><br>'
        'Each pixel beloging to the object must have a <b>unique</b> integer '
        'or RGB(A) value, while background pixels must have 0 or black RGB(A) '
        'value.<br><br>'
        'Allowed <b>file formats</b>: .npy, .npz, .h5, .png, .tif, .tiff, '
        '.jpg, .jpeg, .mov, .avi, and .mp4'
    ),
    'refChFilePath': (f"""
        <b>OPTIONAL</b>: Path of the file with the <b>reference channel
        signal</b>.<br><br>
        Loading the reference channel allows you to choose one or more of
        the following options:
        {html_func.ul(
            'Automatically <b>segment the reference channel</b>'
            f'(see {get_href_tags()["segmRefCh"]} parameter)<br>',

            '<b>Load a segmentation mask</b> for the reference channel'
            f'(see {get_href_tags()["refChSegmFilePath"]} parameter)<br>',

            '<b>Remove spots</b> that are detected <b>outside of the '
            'reference channel mask</b>'
            f'(see {get_href_tags()["keepPeaksInsideRef"]} parameter)<br>',

            '<b>Comparing the spots signal to the reference channel</b> and'
            'keep only the spots that fulfill a specific criteria'
            f'(see {get_href_tags()["filterPeaksInsideRef"]}'
            f'and {get_href_tags()["gopMethod"]} parameters)<br>'
        )}
        Allowed <b>file formats</b>: .npy, .npz, .h5, .png, .tif, .tiff,
        .jpg, .jpeg, .mov, .avi, and .mp4'
    """),
    'refChSegmFilePath': (f"""
        <b>OPTIONAL</b>: Path of the file with the <b>reference channel
        segmentation mask</b>.<br><br>
        The segmentation mask <b>MUST have the same shape</b> as the reference
        channel signal.<br><br>
        If you load a segmentation mask you can then choose to <b>remove
        spots</b> that are detected <b>outside of the reference channel mask</b>
        (see {get_href_tags()["keepPeaksInsideRef"]} parameter).
    """),
    'pixelWidth': ("""
        <b>Physical size</b> of the pixel in the <b>X direction</b> (width).
        The unit is \u00b5m/pixel.<br><br>
        This parameter will be used to <b>convert the size of the resolution
        limited volume</b> into pixels.
    """),
    'pixelHeight': ("""
        <b>Physical size</b> of the pixel in the <b>Y direction</b> (height).
        The unit is \u00b5m/pixel.<br><br>
        This parameter will be used to <b>convert the size of the diffraction
        limited spot</b> into pixels.
    """),
    'voxelDepth': ("""
        <b>Physical size</b> of the voxel in the <b>Z direction</b> (depth),
        i.e., the distance between each z-slice in \u00b5m.<br><br>
        This parameter will be used to <b>convert the size of the diffraction
        limited spot</b> into pixels.
    """),
    'numAperture': ("""
        Numerical aperture of the objective used to acquire the images.<br><br>
        This parameter is used to calculate the <b>radius <code>r</code>
        of the diffraction limited spot</b> according to the
        Abbe diffraction limit formula:<br>
        <code>d</code>
    """),
    'emWavelen': (
        """"""
    ),
    'zResolutionLimit': (
        """"""
    ),
    'yxResolLimitMultiplier': (
        """"""
    ),
    'spotMinSizeLabels': (
        """"""
    ),
    'aggregate': (
        """"""
    ),
    'gaussSigma': (
        """"""
    ),
    'sharpenSpots': (
        """"""
    ),
    'segmRefCh': (
        """"""
    ),
    'keepPeaksInsideRef': (
        """"""
    ),
    'filterPeaksInsideRef': (
        """"""
    ),
    'refChSingleObj': (
        """"""
    ),
    'refChThresholdFunc': (
        """"""
    ),
    'calcRefChNetLen': (
        """"""
    ),
    'spotDetectionMethod': (
        """"""
    ),
    'spotPredictionMethod': (
        """"""
    ),
    'spotThresholdFunc': (
        """"""
    ),
    'gopMethod': (
        """"""
    ),
    'gopLimit': (
        """"""
    ),
    'doSpotFit': (
        """"""
    ),
    'minSpotSize': (
        """"""
    ),
    'maxSpotSize': (
        """"""
    )
    }
    return paramsInfoText
