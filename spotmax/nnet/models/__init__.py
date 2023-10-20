from cellacdc.myutils import check_install_package

from spotmax import is_cli, printl

check_install_package(
    'torch', 
    is_cli=is_cli,
    caller_name='spotMAX'
)

check_install_package(
    'pytorch3dunet', 
    pypi_name='git+https://github.com/ElpadoCan/pytorch3dunet.git',
    is_cli=is_cli,
    caller_name='spotMAX'
)