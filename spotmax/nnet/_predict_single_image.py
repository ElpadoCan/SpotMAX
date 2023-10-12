import os
import yaml
import skimage.io

from spotmax.nnet import transform
from spotmax.nnet.models.nd_model import  Data, Operation, NDModel, Models

from cellacdc.plot import imshow

# Pre-trained model was trained with images scaled to 73 nm pixel size
BASE_PIXEL_SIZE = 73 
INPUT_PIXEL_SIZE = 72.06
REMOVE_HOT_PIXELS = False

nnet_path = os.path.dirname(os.path.abspath(__file__))
config_yaml_path = os.path.join(nnet_path, 'config.yaml')

print('Reading config file...')
# Read config and convert to dict
with open(config_yaml_path, 'r') as f:
    config = yaml.safe_load(f)

data_path = os.path.join(nnet_path, 'data')

img_data_path = os.path.join(data_path, 'single_volume.tiff')
lab_data_path = os.path.join(data_path, 'single_volume_label.tiff')

print('Loading image data...')
img_data = skimage.io.imread(img_data_path)
lab_data = skimage.io.imread(lab_data_path)

imshow(img_data, lab_data)

print('Preprocessing image data...')
scale_factor = INPUT_PIXEL_SIZE/BASE_PIXEL_SIZE

x_transfomer = transform.ImageTransformer(logs=False)
x_transfomer.set_pipeline([
    transform._rescale,
    # transform._opening,
    transform._normalize,
])

transformed_data = x_transfomer.transform(img_data, scale=scale_factor)

input_data = Data(
    images=transformed_data, 
    masks=None, 
    val_images=None, 
    val_masks=None
)

print('Running inference...')
# Predict with 2D model
OPERATION = Operation.PREDICT
MODEL = Models.UNET2D
nd_model = NDModel(
    operation=OPERATION,
    model=MODEL,
    config=config
)
output = nd_model(input_data)

import pdb; pdb.set_trace()