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
        '<b>OPTIONAL</b>: Path of the image file with the '
        '<b>spots channel signal</b>.<br><br>'
        'Leave empty if you only need to segment the reference channel '
        f'(see the {get_href_tags()["refChFilePath"]}).'
        '<br><br>'
        'If you want to analyse multiple Positions/experiments load the '
        'path of the spots channel signal of any of these Positions.'
        '<br><br>'
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
        'If you want to analyse multiple Positions/experiments load the '
        'path of the segmentation file of any of these Positions.'
        '<br><br>'
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
            f'(see the {get_href_tags()["segmRefCh"]} parameter)<br>',

            '<b>Load a segmentation mask</b> for the reference channel'
            f'(see the {get_href_tags()["refChSegmFilePath"]} parameter)<br>',

            '<b>Remove spots</b> that are detected <b>outside of the '
            'reference channel mask</b>'
            f'(see the {get_href_tags()["keepPeaksInsideRef"]} parameter)<br>',

            '<b>Comparing the spots signal to the reference channel</b> and'
            'keep only the spots that fulfill a specific criteria'
            f'(see the {get_href_tags()["filterPeaksInsideRef"]}'
            f'and {get_href_tags()["gopMethod"]} parameters)<br>'
        )}
        If you want to analyse multiple Positions/experiments load the
        path of the reference channel signal of any of these Positions.
        <br><br>
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
        (see the {get_href_tags()["keepPeaksInsideRef"]} parameter).
        <br><br>
        If you want to analyse multiple Positions/experiments load the
        path of the reference channel segmentation mask of any of these Positions.
        <br><br>
    """),
    'pixelWidth': ("""
        <b>Physical size</b> of the pixel in the <b>X-direction</b> (width).
        The unit is \u00b5m/pixel.<br><br>
        This parameter will be used to <b>convert the size of the resolution
        limited volume</b> into pixels.
    """),
    'pixelHeight': ("""
        <b>Physical size</b> of the pixel in the <b>Y-direction</b> (height).
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
    'numAperture': (
        '<b>Numerical aperture</b> of the objective used to acquire the images.<br><br>'
        'This parameter is used to calculate the <b>radius <code>r</code> '
        'of the diffraction limited spot</b> according to the '
        'Abbe diffraction limit\'s formula:'
        r'<math>r = \frac{\lambda}{2NA}</math>'
        'where <code>\u03bb</code> is the emission wavelength '
        'of the fluorescent reporter '
        f'(see the {get_href_tags()["emWavelen"]} parameter).'
    ),
    'emWavelen': (
        '<b>Emission wavelength</b> of the spots signal\'s fluorescent reporter.<br><br>'
        'This parameter is used to calculate the <b>radius <code>r</code> '
        'of the diffraction limited spot</b> according to the'
        'Abbe diffraction limit formula:'
        r'<math>r = \frac{\lambda}{2NA}</math>'
        'where <code>\u03bb</code> is the emission wavelength '
        'of the fluorescent reporter '
        f'(see the {get_href_tags()["numAperture"]} parameter).'
    ),
    'zResolutionLimit': (
        'Rough estimate of the resolution limit of the microscope in the '
        'Z-direction. The unit is \u00b5m<br><br>'
        'This is typically around 2-3 times the diffraction limit, i.e. around '
        '0.8-1 \u00b5m.<br><br>'
        'This parameter will be used as the <b>radius in Z-direction of the '
        'expected volume</b> of the spots.<br><br>'
        '<b>Increase</b> this parameter if spotMAX detects multiple spots within '
        'a single spot.<br><br>'
        '<b>Decrease</b> this parameter if spotMAX detects fewer spots '
        'than expected.'
    ),
    'yxResolLimitMultiplier': (
        'Factor used to <b>multiply the radii of the spots\'s '
        'expected volume</b> in the X- and Y-direction.<br><br>'
        'This parameter is used in conjunction with '
        f'{get_href_tags()["zResolutionLimit"]} to compute the expected spot '
        f'volume.<br><br>'
        'The final volume is displayed on the '
        f'{get_href_tags()["spotMinSizeLabels"]} labels.<br><br>'
        '<b>Increase</b> this parameter if spotMAX detects multiple spots within '
        'a single spot.<br><br>'
        '<b>Decrease</b> this parameter if spotMAX detects fewer spots '
        'than expected.'
    ),
    'spotMinSizeLabels': (
        '<b>Radii</b> in Z-, Y- and X-direction of the '
        '<b>expected spot volume</b>.<br><br>'
        f'The Z-radius is equal to the {get_href_tags()["zResolutionLimit"]} '
        'parameter.<br><br>'
        'The X- and Y- radii are computed from the <b>Abbe diffraction limit\'s '
        'formula</b> (see below). The result of this formula is then multiplied '
        f'by {get_href_tags()["yxResolLimitMultiplier"]} to obtain the radii '
        'in \u00b5m.<br><br>'
        'Finally, the radii are converted from \u00b5m into pixels using the '
        f'{get_href_tags()["pixelWidth"]}, '
        f'the {get_href_tags()["pixelHeight"]}, and the '
        f'{get_href_tags()["voxelDepth"]}<br><br>'
        'Abbe diffraction limit formula:<br>'
        r'<math>r = \frac{\lambda}{2NA}</math>'
        f'where <code>\u03bb</code> is the {get_href_tags()["emWavelen"]} '
        f'and <code>NA</code> is the {get_href_tags()["numAperture"]}.'
    ),
    'aggregate': (
        'If you chose to aggreate the objects, spotMAX will first create '
        'an image with the spots <b>signal from each '
        'bounding box stacked next to each other</b>.<br><br>'
        'The segmentation mask can be loaded from the '
        f'{get_href_tags()["segmFilePath"]} parameter.<br><br>'
        'Activate this option if you expect some cells in the image to be '
        'without any spot. In this case, aggregation is required because '
        '<b>automatic thresholding</b> will be <b>more accurate</b> '
        'if applied to aggreagated objects.<br><br>'
        'Note that this parameter is <b>disabled if you do not load '
        'any segmentation file.</b>'
    ),
    'gaussSigma': (
        'Sigma of the gaussian filter applied during pre-processing. Write 0 '
        'if you do not want any gaussian filter.<br><br>'
        'This step usually <b>greatly helps reducing noise</b>, '
        'however we recommend to apply a guassian filter with '
        'a small sigma &#60; 1.<br><br>'
        'Note that this affect only the detection. <b>Quantification is '
        'performed only on raw signal</b>.'
    ),
    'sharpenSpots': (
        'Choose whether to apply a <b>sharpening filter</b> to improve the signal '
        'to noise ratio.<br><br>'
        'We recommend applying this filter, however, you can try to '
        'deactivate it if spotMAX detects more spots than expected.<br><br>'
        'The filtered image is the result of the <b>subtraction</b> between '
        'the <b>raw image</b> and a <b>blurred image</b>. The blurred image '
        'is obtained with a gaussian filter with one sigma for each dimension '
        'and values equal to the radii of the diffraction limited volume (see '
        f'{get_href_tags()["spotMinSizeLabels"]}).'
    ),
    'segmRefCh': (
        'Choose whether to <b>automatically segment</b> the reference '
        'channel signal.<br><br>'
        'Once segmented spotMAX can remove spots that are outside of the '
        'reference channel mask (only if you also activate '
        f'{get_href_tags()["keepPeaksInsideRef"]} parameter)'

    ),
    'keepPeaksInsideRef': (
        'If you activate this option AND provide a segmentation mask '
        'for the reference channel '
        f'(see {get_href_tags()["refChSegmFilePath"]}) '
        'or you load AND segment the reference channel '
        f'(see {get_href_tags()["segmRefCh"]}) '
        'spotMAX will then <b>remove the spots that are detected outside '
        'of the reference channel mask</b>.'
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
