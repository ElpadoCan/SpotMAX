import os
import yaml
import skimage.io

from spotmax.nnet import config_yaml_path, data_path
from spotmax.nnet.model import Model

from cellacdc.plot import imshow

def main():
    # Pre-trained model was trained with images scaled to 73 nm pixel size
    img_data_path = os.path.join(data_path, 'single_volume.tiff')
    lab_data_path = os.path.join(data_path, 'single_volume_label.tiff')

    print('Loading image data...')
    img_data = skimage.io.imread(img_data_path)[:, 235:367, 465:638]
    lab_data = skimage.io.imread(lab_data_path)[:, 235:367, 465:638]

    print(f'Image shape = {img_data.shape}')
    imshow(img_data, lab_data)
    import pdb; pdb.set_trace()

    model = Model(
        model_type='3D',
        preprocess_across_experiment=False, 
        preprocess_across_timepoints=False, 
        remove_hot_pixels=True,
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

if __name__ == '__main__':
    main()