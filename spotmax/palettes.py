from cellacdc._palettes import get_color_scheme

def ini_hex_colors():
    scheme = get_color_scheme()
    if scheme == 'light':
        section = '#8000ff'
        option = '#0000ff'
    else:
        section = '#8ccfd3'
        option = '#dfc37d'
    ini_palette = {
        'section': section, 
        'option': option
    }
    return ini_palette