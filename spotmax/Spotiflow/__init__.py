from cellacdc.myutils import check_install_package

from spotmax import is_cli

def install():
    check_install_package(
        'spotiflow', 
        is_cli=is_cli,
        caller_name='SpotMAX',
        installer='conda'
    )