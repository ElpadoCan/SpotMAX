from pathlib import Path

import numpy as np
import torch

from bioimageio.spec.model.v0_5 import (
    Author,
    Maintainer,
    AxisId,
    BatchAxis,
    ChannelAxis,
    CiteEntry,
    EnvironmentFileDescr,
    DatasetId,
    Doi,
    FileDescr,
    HttpUrl,
    Identifier,
    InputTensorDescr,
    IntervalOrRatioDataDescr,
    LicenseId,
    LinkedDataset,
    ModelDescr,
    OrcidId,
    ParameterizedSize,
    PytorchStateDictWeightsDescr,
    ScaleRangeDescr,
    ScaleRangeKwargs,
    SpaceInputAxis,
    TensorId,
    TorchscriptWeightsDescr,
    WeightsDescr,
    ArchitectureFromFileDescr,
    Version
)
from bioimageio.spec.pretty_validation_errors import (
    enable_pretty_validation_errors_in_ipynb,
)

from spotmax import spotmax_path


"""Model Input"""
print('Loading model input...')
model_folder_root = Path(spotmax_path) / 'BioImageIO' / 'SpotMAX_UNet_2D'
input_sample_path = model_folder_root / 'input_sample.npy'
input_sample = np.load(str(input_sample_path))
Z, Y, X = input_sample.shape[-3:]
print(Z, Y, X, input_sample.dtype)
import pdb; pdb.set_trace()

input_descr = InputTensorDescr(
    id=TensorId('spots'),
    axes=[
        BatchAxis(),
        SpaceInputAxis(
            id=AxisId('z'),
            size=43,
            concatenable=False
        ),
        SpaceInputAxis(
            id=AxisId('y'),
            size=123,
            concatenable=False
        ),
        SpaceInputAxis(
            id=AxisId('x'),
            size=167,
            concatenable=False),
    ],
    test_tensor=FileDescr(source=input_sample_path),
    # sample_tensor=FileDescr(source=input_sample_path),
    data=IntervalOrRatioDataDescr(type='float32'),
)

"""Model Output"""
print('Loading model output...')
output_sample_path = model_folder_root / 'output_sample_mask.npy'
output_sample = np.load(str(output_sample_path))
Z, Y, X = output_sample.shape[-3:]
print(Z, Y, X, output_sample.dtype)
import pdb; pdb.set_trace()

from bioimageio.spec.model.v0_5 import (
    OutputTensorDescr, SizeReference, SpaceOutputAxis
)

output_descr = OutputTensorDescr(
    id=TensorId('mask'),
    description='predicted boolean mask of spot areas',
    axes=[
        BatchAxis(),
        # ChannelAxis(channel_names=[Identifier('prediction')]),
        SpaceOutputAxis(id=AxisId('z'), size=43),
        SpaceOutputAxis(id=AxisId('y'), size=123),
        SpaceOutputAxis(id=AxisId('x'), size=167),
    ],
    test_tensor=FileDescr(source=output_sample_path),
    data=IntervalOrRatioDataDescr(type='uint8'),
)

"""Model Architecture"""
print('Generating model architecture...')
model_py_path = Path(spotmax_path) / 'nnet' / 'model.py'

pytorch_version = Version(torch.__version__)

pytorch_architecture = ArchitectureFromFileDescr(
    source=model_py_path,
    callable=Identifier('Model'),
    kwargs=dict(
        model_type='2D', 
        preprocess_across_experiment=False,
        preprocess_across_timepoints=False,
        gaussian_filter_sigma=0,
        remove_hot_pixels=True,
        config_yaml_filepath='./config.yaml', 
        PhysicalSizeX=0.06725,
        resolution_multiplier_yx=1, 
        use_gpu=True, 
        save_prediction_map=False, 
        verbose=False,
    )
)
print(model_py_path)
import pdb; pdb.set_trace()

"""Create model description"""
print('Creating model description...')
model_descr = ModelDescr(
  name='SpotMAX-AI 2D',
  description=(
    'SpotMAX - 2D fluorescence spot segmentation'
  ),
  covers=[model_folder_root / 'cover.png'],
  authors=[
      Author(
        name='Francesco Padovani',
        affiliation='Helmholtz Munich',
        email='padovaf@tcd.ie',
        github_user='ElpadoCan',
        orcid=OrcidId('0000-0003-2540-8240')
      )
  ],
  maintainers=[
    Maintainer(
      name='Francesco Padovani',
      affiliation='Helmholtz Munich',
      email='padovaf@tcd.ie',
      github_user='ElpadoCan',
      orcid=OrcidId('0000-0003-2540-8240')
    )
  ],
  cite=[
    CiteEntry(
      text=(
        'Padovani, F., Čavka, I., Neves, A. R. R., López, C. P., Al-Refaie, N., '
        'Bolcato, L., Chatzitheodoridou, D., Chadha, Y., Su, X.A., Lengefeld, J., '
        'Cabianca D. S., Köhler, S., Schmoller, K. M. SpotMAX: a generalist '
        'framework for multi-dimensional automatic spot detection and quantification, '
        'bioRxiv (2024) DOI: 10.1101/2024.10.22.619610'
      ),
      doi=Doi('10.1101/2024.10.22.619610'),
    )
  ],
  license=LicenseId('GPL-3.0-only'),
  documentation=model_folder_root / 'README.md',
  git_repo=HttpUrl('https://github.com/ElpadoCan/SpotMAX'),
  links=[
    HttpUrl('https://spotmax.readthedocs.io/en/latest/')
  ],
  tags=[
      'spot-detection',
      'diffraction-limited-spots',
      'pytorch',
      'fluorescence-light-microscopy',
      'spotmax',
  ],
  # training_data=LinkedDataset(id=DatasetId('uplifting-ice-cream')),
  inputs=[input_descr],
  outputs=[output_descr],
  weights=WeightsDescr(
      pytorch_state_dict=PytorchStateDictWeightsDescr(
          source=model_folder_root / 'unet_best.pth',
          architecture=pytorch_architecture,
          pytorch_version=pytorch_version,
          dependencies=EnvironmentFileDescr(
            source=model_folder_root / 'environment.yml'
          )
      ),
  ),
  attachments=[
    FileDescr(source=model_folder_root / 'model_usage.py'),
    FileDescr(source=model_folder_root / 'config.yaml'),
  ],
)
import pdb; pdb.set_trace()

"""Test the model"""
print('Testing the model...')
from bioimageio.core import test_model

validation_summary = test_model(model_descr)
print(validation_summary.display())

answer = input('Do you want to proceed to package the model in a zip file? ([y]/n): ')
if answer.lower() == 'n':
    exit('Execution stopped. Zip archive for the model resource not created.')

"""Package the model

Save all the model files to a zip file that can be uploaded to BioImage.IO
"""
from pathlib import Path

from bioimageio.spec import save_bioimageio_package

output_path = save_bioimageio_package(
    model_descr, 
    output_path=model_folder_root / 'SpotMAX_AI_2D.zip'
)
print('*'*100)
print(f'Done. Model ZIP file path: {output_path}')