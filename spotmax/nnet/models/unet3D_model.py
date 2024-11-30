import importlib
from .base_model import BaseModel
import numpy as np
from .unet3D.unet3d import utils
from .unet3D.datasets.utils import get_test_numpy_loader
from .unet3D.unet3d.model import get_model
from .unet3D.unet3d.trainer import create_trainer
import os
import torch


def _get_predictor(model, output_dir, config):
    predictor_config = config.get('predictor', {})
    class_name = predictor_config.get('name', 'StandardPredictor')

    m = importlib.import_module('spotmax.nnet.models.unet3D.unet3d.predictor')
    predictor_class = getattr(m, class_name)

    return predictor_class(model, output_dir, config, **predictor_config)



class Unet3DModel(BaseModel):

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        # Get a device to train on
        device_str = self.config.get('device', 'cpu')        
        device = torch.device(device_str)

        # Set the device in the config
        self.config['device'] = device      
        
        self.model = None
        try:
            model = get_model(self.config['model'])
            model_path = os.path.expanduser(self.config['model_path'])
            utils.load_checkpoint(model_path, model)      
            self.model = model
        except Exception as err:
            pass

    def train(
            self,
            X_train,
            y_train,
            X_val,
            y_val
        ):
        """Train the model."""

        # Adding the data to the config, so we can use them in the trainer
        self.config['loaders']['X_train'] = X_train
        self.config['loaders']['y_train'] = y_train
        self.config['loaders']['X_val'] = X_val
        self.config['loaders']['y_val'] = y_val

        # create trainer
        trainer = create_trainer(self.config)
        # Start training
        trainer.fit()

    def predict(self, images: np.ndarray) -> np.ndarray:
        """Predict the model."""

        if self.model is None:
            # Read the model trained
            model = get_model(self.config['model'])
            model_path = os.path.expanduser(self.config['model_path'])
            utils.load_checkpoint(model_path, model)
        else:
            model = self.model

        # Moving the model to the GPU
        model = model.to(self.config['device'])
        output_dir = self.config['loaders'].get('output_dir', None)

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        # create predictor instance
        predictor = _get_predictor(model, output_dir ,self.config)
        test_loader = get_test_numpy_loader(self.config, images)
        
        # Run inference
        predicted = predictor(test_loader)
        
        # To numpy
        predictions = np.asarray(predicted).squeeze()

        # if predictions have 2 dimensions reshape to have 3D array
        if len(predictions.shape) == 2:
            predictions = np.expand_dims(predictions, axis=0)

        return predictions

