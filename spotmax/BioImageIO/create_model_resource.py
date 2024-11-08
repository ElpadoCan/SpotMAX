"""
This script is used to create a BioImage.IO model ready to be uploaded on the 
model zoo. 

It requires the `cellacdc`, `spotmax`, and `bioimageio.spec` package. The 
`bioimageio.spec` package will be automatically installed if missing. 
"""

from cellacdc import myutils

myutils.check_install_package(
    'bioimageio.spec', is_cli=True, caller_name='SpotMAX'
)

from bioimageio.spec.model.v0_5 import ModelDescr
from bioimageio.spec.model.v0_5 import (
    AxisId,
    BatchAxis,
    ChannelAxis,
    FileDescr,
    Identifier,
    InputTensorDescr,
    IntervalOrRatioDataDescr,
    ParameterizedSize,
    SpaceInputAxis,
    SpaceOutputAxis,
    TensorId,
    WeightsDescr,
)


