from functools import wraps
import re

from . import is_mac

def _tag(tag_info='p style="font-size:10px"'):
    def wrapper(func):
        @wraps(func)
        def inner(text):
            tag = tag_info.split(' ')[0]
            text = f'<{tag_info}>{text}</{tag}>'
            return text
        return inner
    return wrapper

@_tag(tag_info='i')
def italic(text):
    return text

@_tag(tag_info='b')
def bold(text):
    return text

def untag(text, tag):
    """Extract texts from inside an html tag and outside the html tag

    Parameters
    ----------
    text : str
        Input text.
    tag : str
        Name of the html tag, e.g., 'p' or 'a'.

    Returns
    -------
    tuple
        Tuple of two lists. One list with texts from inside the tags
        and one list with texts from outside the tags.

    """
    start_tag_iter = re.finditer(f'<{tag}.*?>', text)
    stop_tag_iter = re.finditer(f'</{tag}>', text)

    in_tag_texts = []
    out_tag_texts = []
    prev_i1_close_tag = 0
    i1_close_tag = len(text)-1
    for m1, m2 in zip(start_tag_iter, stop_tag_iter):
        i0_open_tag, i1_open_tag = m1.span()
        i0_close_tag, i1_close_tag = m2.span()
        end = m2.span()[0]
        in_tag_text = text[i1_open_tag:i0_close_tag]
        in_tag_texts.append(in_tag_text)
        out_tag_text = text[prev_i1_close_tag:i0_open_tag]
        out_tag_texts.append(out_tag_text)
        prev_i1_close_tag = i1_close_tag

    out_tag_text = text[i1_close_tag:]
    out_tag_texts.append(out_tag_text)
    in_tag_texts.append('')

    return in_tag_texts, out_tag_texts

def tag(text, tag_info='p style="font-size:10pt"'):
    tag = tag_info.split(' ')[0]
    text = f'<{tag_info}>{text}</{tag}>'
    return text

def href(text, link):
    return f'<a href="{link}">{text}</a>'

def span(text, font_color=None):
    if font_color is not None:
        s = (
            f'<span style="color:{font_color};">'
            f'{text}'
            '</span>'
        )
    else:
        s = (f'<span>{text}</span>')
    return s

def paragraph(txt, font_size='13px', font_color=None, wrap=True, center=False):
    if not wrap:
        txt = txt.replace(' ', '&nbsp;')
    if font_color == 'r':
        font_color = '#FF0000'
    if font_color is None:
        s = (f"""
        <p style="font-size:{font_size};">
            {txt}
        </p>
        """)
    else:
        s = (f"""
        <p style="font-size:{font_size}; color:{font_color}">
            {txt}
        </p>
        """)
    if center:
        s = re.sub(r'<p style="(.*)">', r'<p style="\1; text-align:center">', s)
    return s

def ul(*items):
    txt = ''
    for item in items:
        txt = f"{txt}{tag(item, tag_info='li')}"
    return tag(txt, tag_info='ul')

def simple_rst_to_html(rst_text):
    valid_chars = r'[A-Za-z0-9\-\.=_ ]'
    html_text = rst_text.strip('\n')
    html_text = html_text.replace('\n', '<br>')
    html_text = html_text.replace(' <br>', '<br>')
    html_text = re.sub(rf'``({valid_chars}+)``', r'<code>\1</code>', html_text)
    html_text = re.sub(rf'\*\*({valid_chars}+)\*\*', r'<b>\1</b>', html_text)
    html_text = re.sub(rf'\*({valid_chars}+)\*', r'<i>\1</i>', html_text)
    html_text = re.sub(rf'`({valid_chars}+)`_', r'<b>\1</b>', html_text)
    return html_text

if __name__ == '__main__':
    text = 'ciao'
    print(paragraph(text))
    print(italic(text))
    print(bold(text))
    print(tag(text))
