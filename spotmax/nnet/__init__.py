import os

nnet_path = os.path.dirname(os.path.abspath(__file__))
config_yaml_path = os.path.join(nnet_path, 'config.yaml')

data_path = os.path.join(nnet_path, 'data')

from cellacdc.myutils import check_install_package, check_install_torch

from spotmax import is_cli, printl, io

def install_and_download():
    check_install_torch(is_cli=is_cli, caller_name='SpotMAX')

    check_install_package(
        'pytorch3dunet', 
        pypi_name='pytorch3dunet-spotmax', # 'git+https://github.com/ElpadoCan/pytorch3dunet.git',
        is_cli=is_cli,
        caller_name='SpotMAX'
    )

    check_install_package(
        'yaml', 
        pypi_name='pyyaml',
        is_cli=is_cli,
        caller_name='SpotMAX'
    )

    io.download_unet_models()