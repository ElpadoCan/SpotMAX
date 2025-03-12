.. _ai_params:

SpotMAX AI parameters
=====================

Here you can find the description of the paramters required to configure 
and run the neural network models provided with SpotMAX. 

You can set this paramaters from the GUI (see the 
:confval:`Spots segmentation method` parameter) or manually set them in the 
SpotMAX INI configuration file. At the end of this page, you can find an example 
of the INI configuration file with the AI parameters.

.. confval:: Model type

    Model type to use for the prediction. The available models are: ``2D`` and ``3D``. 
    The default value is ``2D``. 
    If you choose the 2D model and the input data is 3D z-stacks, then the 
    prediction will be done slice by slice.

    :type: string ``2D`` or ``3D``
    :default: ``2D``

.. confval:: Preprocess across experiment

    This paramater has two different behaviours depending on how you are running 
    the model. More details about preprocessing below. 
    
    If you are running the model as part of the entire SpotMAX workflow (from GUI 
    or from the command line with the ``spotmax -p`` command) and you set this parameter 
    to ``True``, the model will run preprocessing across all Positions in each 
    experiment folder. Setting this paramater to ``True`` will result 
    in using the global minimum and maximum intensity values across all Positions 
    in the experiment to normalize the images to the range [0, 1].
    
    If you are running the model from the Python APIs and you set this parameter 
    to ``True``, the model will run preprocessing on all input images. If you 
    set it to ``False``, no preprocessing will be applied **only if also** 
    :confval:`Preprocess across timepoints` is ``False``. 

    .. note:: 
        
        Preprocessing pipeline includes the following steps:

            1. Morphological opening to remove hot pixels (isolated bright pixels). 
               This step is applied only if :confval:`Remove hot pixels` is ``True``.
            2. Gaussian filter (smooth) using the sigma value defined in the 
               :confval:`Gaussian filter sigma` parameter. This step is applied 
               only if :confval:`Gaussian filter sigma` is greater than 0.0.
            3. Normalization of the input images to the range [0, 1] using the 
               minimum and maximum intensity values across the timepoints of each 
               video (if :confval:`Preprocess across timepoints` is ``True``) or 
               using the global minimum and maximum intensity values across all 
               Positions in the experiment (if :confval:`Preprocess across experiment` 
               is ``True``).

    :type: boolean
    :default: ``False``

.. confval:: Preprocess across timepoints

    This paramater has two different behaviours depending on how you are running 
    the model. More details about preprocessing below. 
    
    If you are running the model as part of the entire SpotMAX workflow (from GUI 
    or from the command line with the ``spotmax -p`` command) and you set this parameter 
    to ``True``, the model will run preprocessing across all timepoints of each 
    loaded Position. Setting this paramater to ``True`` will result 
    in using the minimum and maximum intensity values across the timepoints of each 
    video to normalize the images to the range [0, 1].
    
    If you are running the model from the Python APIs (using the ``Model`` class 
    from ``spotmax.nnet.model``) and you set this parameter 
    to ``True``, the model will run preprocessing on all input images 
    (no matter if they are timepoints or not). If you set it to ``False``, 
    no preprocessing will be applied **only if also** 
    :confval:`Preprocess across experiment` is ``False``. 

    .. note:: 
        
        See :confval:`Preprocess across experiment` parameter for more 
        details about preprocessing.

    :type: boolean
    :default: ``True``

.. confval:: Gaussian filter sigma

    Sigma of the Gaussian filter (smoothing fitler) to apply to the input images. 
    This can be a  single float value or a sequence of three floats (z, y, x) 
    to apply different  sigmas in each direction. 
    
    If the input images are 2D, the sigma value in the z-direction will be ignored.

    Default value is 0.0, which means no smoothing will be applied.

    :type: float or (z, y, x) sequence of floats
    :default: ``0.0``


.. confval:: Remove hot pixels

    If ``True``, the model will apply a morphological opening filter to remove 
    hot pixels (isolated bright pixels) from the input images.

    :type: boolean
    :default: ``False``

.. confval:: Config YAML filepath

    Path to the configuration file with the model parameters. This is created 
    during training. The default value is ``spotmax/nnet/config.yaml``, 
    which are the pretrained models provided with SpotMAX.

    You can view this file `here <https://github.com/SchmollerLab/SpotMAX/blob/main/spotmax/nnet/config.yaml>`_. 

    :type: PathLike
    :default: ``spotmax/nnet/config.yaml``

.. confval:: Threshold value

    Threshold value to apply to the prediction map. The default value is 0.0, 
    meaning that the value will be retrieved from the configuration file.

    The value for the pre-trained 2D model is 0.9, and for the 3D model is 0.7. 

    :type: float
    :default: ``0.0``

.. confval:: PhysicalSizeX

    Physical size of a pixel in the x-direction in micrometers. This value is 
    used in conjuction with the :confval:`Resolution multiplier YX` to calculate 
    the scaling factor for the input images. 

    The scaling factor is used to resize the input images to the same pixel size 
    of the training images.

    The default value is 0.073, meaning that if :confval:`Resolution multiplier YX` 
    is 1.0, and you are using the pretrained models, the input images will not 
    be resized.

    The scaling factor :math:`S_f` is calculated as follows:

    .. math::

        S_f = p_{input} \cdot \frac{1}{p_{training}} \cdot \frac{1}{r_{yx}}
    
    where :math:`p_{input}` is defined in the :confval:`PhysicalSizeX` parameter, 
    :math:`p_{training}` is the physical size of a pixel in the training images 
    (defined in the configuration file at the ``base_pixel_size_nm``), 
    and :math:`r_{yx}` is defined in the :confval:`Resolution multiplier YX` parameter.

    Note that the same scaling factor is then used to rescale the model output 
    to the original size of the input images.

    :type: float
    :default: ``0.073``

.. confval:: Resolution multiplier YX

    Additional factor to reduce the scaling factor 
    (see :confval:`PhysicalSizeX` for more information) when resizing the 
    input images. Pass a value greater than 1.0 when you need to detect spots that 
    are larger than the diffraction limit. 
    
    Default is 1.0

    :type: float
    :default: ``1.0``

.. confval:: Use GPU

    If ``True``, the model will use the GPU to run the prediction. Make sure 
    you have a compatible GPU and the required libraries installed 
    (e.g., NVIDIA GPU with PyTorch and CUDA drivers).

    :type: boolean
    :default: ``False``

.. confval:: Save prediction map

    If ``True``, the model will either return the prediction map or save it 
    as a NPZ file in the same directory as the input images. 

    The prediction map is returned when you are running the model 
    from the Python APIs (using the ``Model.segment`` method from ``spotmax.nnet.model``), 
    while it is saved as a NPZ file when you are running the model as part of 
    the entire SpotMAX workflow (from GUI or from the command line with the 
    ``spotmax -p`` command).

    :type: boolean
    :default: ``False``

.. confval:: Label components

    If ``True``, the output boolean masks will be converted to connected 
    components using the ``skimage.measure.label`` function. 

    More information `here <https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.label>`_.

    :type: boolean
    :default: ``False``

Example of the INI configuration file with the AI parameters

.. code-block:: ini

    [Spots channel]
    Spots segmentation method = spotMAX AI

    [neural_network.init.spots]
    model_type = 3D
    preprocess_across_experiment = False
    preprocess_across_timepoints = True
    gaussian_filter_sigma = 0.0
    remove_hot_pixels = False
    config_yaml_filepath = spotmax/nnet/config.yaml
    threshold_value = 0.7
    PhysicalSizeX = 0.073
    resolution_multiplier_yx = 1.0
    use_gpu = False
    save_prediction_map = False
    verbose = True

    [neural_network.segment.spots]
    label_components = False