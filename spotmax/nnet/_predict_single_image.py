import os
import yaml
import skimage.io

from spotmax.nnet import config_yaml_path, data_path
from spotmax.nnet import transform
from spotmax.nnet.model import Model
from spotmax.nnet.models.nd_model import  Data, Operation, NDModel, Models

from cellacdc.plot import imshow

# Pre-trained model was trained with images scaled to 73 nm pixel size
BASE_PIXEL_SIZE = 73 
INPUT_PIXEL_SIZE = 72.06
REMOVE_HOT_PIXELS = False

print('Reading config file...')
# Read config and convert to dict
with open(config_yaml_path, 'r') as f:
    config = yaml.safe_load(f)
    
img_data_path = os.path.join(data_path, 'single_volume.tiff')
lab_data_path = os.path.join(data_path, 'single_volume_label.tiff')

print('Loading image data...')
img_data = skimage.io.imread(img_data_path)
lab_data = skimage.io.imread(lab_data_path)

imshow(img_data, lab_data)

model = Model(
    model_type='2D',
    preprocess_across_experiment=False, 
    config_yaml_filepath=config_yaml_path,
    PhysicalSizeX=0.07206,
    use_gpu=True
)
thresholded = model.segment(img_data)

# print('Preprocessing image data...')
# scale_factor = INPUT_PIXEL_SIZE/BASE_PIXEL_SIZE

# x_transfomer = transform.ImageTransformer(logs=False)
# x_transfomer.set_pipeline([
#     transform._rescale,
#     # transform._opening,
#     transform._normalize,
# ])

# transformed_data = x_transfomer.transform(img_data, scale=scale_factor)

# input_data = Data(
#     images=transformed_data, 
#     masks=None, 
#     val_images=None, 
#     val_masks=None
# )

# print('Running inference...')
# # Predict with 2D model
# OPERATION = Operation.PREDICT
# MODEL = Models.UNET2D
# nd_model = NDModel(
#     operation=OPERATION,
#     model=MODEL,
#     config=config
# )
# prediction, threshold_value = nd_model(input_data)

imshow(
    img_data, lab_data, thresholded,
    axis_titles=['Raw image', 'Ground truth', 'Binary prediction']
)