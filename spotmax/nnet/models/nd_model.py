from dataclasses import dataclass
import numpy as np
from enum import Enum
from .unet2d_model import Unet2DModel
from .unet3D_model import Unet3DModel

from .. import printl

from pprint import pprint

@dataclass
class Data:
    """
    This class is used to store the data for the model.
    Images and masks should be in the shape (x, y, z) or (x, y).
    Which corresponds to (x, y, z) or (x, y).
    """
    images: np.ndarray
    masks: np.ndarray
    val_images: np.ndarray
    val_masks: np.ndarray

    def check_dimensions(self):
        assert self.images.shape == self.masks.shape
        assert self.val_images.shape == self.val_masks.shape if self.val_images is not None else True
        assert self.images.ndim in [2, 3]
        assert self.masks.ndim in [2, 3] if self.masks is not None else True
        assert self.val_images.ndim in [2, 3] if self.val_images is not None else True
        assert self.val_masks.ndim in [2, 3] if self.val_masks is not None else True

class Operation(Enum):
    """Enum for the operations."""

    TRAIN = 'train'
    PREDICT = 'predict'

class Models(Enum):
    """Enum for the models."""
    UNET2D = 'unet2D'
    UNET3D = 'unet3D'

models = {
    Models.UNET2D: Unet2DModel,
    Models.UNET3D: Unet3DModel
}

default_threshold = {
    Models.UNET2D: 0.9,
    Models.UNET3D: 0.7,
}

def check_valid_operation(operation:Operation, model:Models, data:Data):
    """Check if the operation is valid for the model.

    Args:
        operation (Operation): The operation to perform (train or predict).
        model (Models): The model to use (2D, 3D).
        data (Data): The data to use. If the operation is train, the data should contain validation data.

    Raises:
        ValueError: If the operation is not valid for the model.
        ValueError: If the model is not valid.
        ValueError: If the operation is train and the data does not contain validation data.
        ValueError: If the operation is predict and the data contains validation data.
    """
    if operation not in Operation:
        raise ValueError(f'Invalid operation: {operation}')
    if model not in Models:
        raise ValueError(f'Invalid model: {model}')
    if operation == Operation.TRAIN:
        if data.images is None or data.masks is None:
            raise ValueError('Training data is not provided')
        if data.val_images is None or data.val_masks is None:
            raise ValueError('Validation data is not provided')
    if operation == Operation.PREDICT:
        if data.images is None:
            raise ValueError('Image is not provided')

class NDModel(object):
    """ Model class for wrapping 2D and 3D models."""

    def __init__(self, operation:Operation, model:Models, config):
        """Initialize the model.

        Args:
            operation (Operation): The operation to perform (train or predict).
            model (Models): The model to use (2D, 3D).
            config (_type_): The config to use.
        """
        self.operation = operation
        self.model = model
        self.config = config

    def _init_model_instance(self, verbose=None):
        if self.model == Models.UNET2D:
            config = self.config[self.model.value]
        else:
            config = self.config[self.model.value][self.operation.value]

        if verbose is None:
            verbose = self.config.get('verbose', True)
            
        config['verbose'] = verbose
        
        # Print operation and configuration of the model
        if config['verbose']:
            config_str = '\n'.join(
                [f'{key}: {value}' for key, value in config.items()]
            )
            print(
                "########################################\n"
                "######## Model Configuration ###########\n"
                "########################################\n"
                f"Model: {self.model.value}\n"
                f"Operation: {self.operation.value}\n"
                f"Configuration:\n"
                f"{config_str}\n"
                "########################################"
            )
        
        config['device'] = self.config['device']
        
        # Instanciate the model using the config
        model_instance = models[self.model](config)
        
        return model_instance
    
    def __call__(self, data:Data, verbose=True):
        """Call the model and perform the operation.

        Args:
            data (Data): The data to use. If the operation is train, the data should contain validation data.

        Returns:
            None or Tuple: If the operation is train, return None.
            If the operation is predict, return the prediction and the threshold.
        """

        # Check inputs depedning on the operation
        # data.check_dimensions()
        check_valid_operation(self.operation, self.model, data)
        
        # Instanciate the model using the config
        model_instance = self._init_model_instance()

        # Train or predict
        if self.operation == Operation.TRAIN:
            model_instance.train(
                X_train=data.images,
                y_train=data.masks,
                X_val=data.val_images,
                y_val=data.val_masks,
            )
        if self.operation == Operation.PREDICT:
            predictions = model_instance.predict(data.images)
            return predictions, default_threshold[self.model]
