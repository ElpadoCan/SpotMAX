from functools import wraps
import re

from cellacdc._palettes import (
    _get_highligth_header_background_rgba, _get_highligth_text_background_rgba
)
from cellacdc.colors import rgb_uint_to_html_hex

import cellacdc.html_utils as acdc_html

RST_NOTE_DIR_RGBA = _get_highligth_header_background_rgba()
RST_NOTE_DIR_HEX_COLOR = rgb_uint_to_html_hex(RST_NOTE_DIR_RGBA[:3])

RST_NOTE_TXT_RGBA = _get_highligth_text_background_rgba()
RST_NOTE_TXT_HEX_COLOR = rgb_uint_to_html_hex(RST_NOTE_TXT_RGBA[:3])

ADMONITION_TYPES = (
    'topic', 
    'admonition', 
    'attention', 
    'caution', 
    'danger', 
    'error', 
    'hint', 
    'important', 
    'note', 
    'seealso', 
    'tip', 
    'todo', 
    'warning', 
    'versionadded', 
    'versionchanged', 
    'deprecated'
)

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

def to_admonition(*args, **kwargs):
    return acdc_html.to_admonition(*args, **kwargs)

def tag(text, tag_info='p style="font-size:10pt"'):
    tag = tag_info.split(' ')[0]
    text = f'<{tag_info}>{text}</{tag}>'
    return text

def href(text, link):
    return f'<a href="{link}">{text}</a>'

def span(text, font_color=None, background_color=None):
    if font_color is not None:
        open_tag = f'<span style="color:{font_color};">'
        if background_color is not None:
            open_tag = open_tag.replace(
                ';">', f'; background-color:{font_color};">'
            )
    elif background_color is not None:
        open_tag = f'<span style="background-color:{background_color};">'
    else:
        open_tag = '<span>'
    
    s = (f'{open_tag}{text}</span>')
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

def get_indented_paragraph(rst_text):
    lines = rst_text.split('\n')
    if not lines:
        return rst_text
    
    indentation = ''
    for i, line in enumerate(lines):
        if not line:
            continue
        
        if not line.strip():
            continue
        
        if not indentation:
            lstripped = line.lstrip()
            end_indent = line.find(lstripped)
            indentation = line[:end_indent]
        
        if not line.startswith(indentation):
            break
    else:
        i = len(lines)
    
    indented_lines = lines[:i]
    indented_paragraph = '\n'.join(indented_lines)
    return indented_paragraph, indentation

def rst_admonitions_to_html_table(rst_text, admonition_type):
    rst_with_html_admons = rst_text
    
    admonition_type = admonition_type.lower()
    
    while True:        
        rst_admon_with_dir, indented_paragraph = rst_extract_directive_block(
            rst_text, f'.. {admonition_type}::'
        )
        if not rst_admon_with_dir:
            break
        
        # Remove multiple spacing with single one
        html_paragraph = re.sub(r' +', ' ', indented_paragraph)
        
        # Remove trailing spaces and new lines chars
        html_paragraph = html_paragraph.strip('\n')
        html_paragraph = html_paragraph.strip()
        
        # Replace new line chars with <br>
        html_paragraph = html_paragraph.replace('\n', '<br>')
        html_paragraph = html_paragraph.replace('<br> ', '<br>')
        
        html_admon = acdc_html.to_admonition(
            html_paragraph, admonition_type=admonition_type
        )
        
        # Replace rst note with html note
        rst_with_html_admons = rst_with_html_admons.replace(
            rst_admon_with_dir, html_admon
        )
        
        # Remove current admon block
        end_current_block_idx = (
            rst_with_html_admons.find(rst_admon_with_dir) 
            + len(rst_admon_with_dir)
        )
        rst_text = rst_text[end_current_block_idx:]
        
    return rst_with_html_admons

def rst_extract_directive_block(rst_text, directive):
    dir_text = rst_text
    
    # Find first dir directive
    dir_idx = dir_text.find(directive)
    if dir_idx == -1:
        return '', ''
    
    # Get everything after dir directive
    text_with_dir = dir_text[dir_idx:]
    dir_text = dir_text[dir_idx+len(directive):]
    
    # Find next dir directive
    next_dir_idx = dir_text.find(directive)
    if next_dir_idx == -1:
        next_dir_idx = len(dir_text)
    
    # Get the text of the dir (indented paragraph)
    indented_paragraph, indentation = get_indented_paragraph(dir_text)
    
    # Exctract dir directive with its text
    end_paragraph_idx = (
        text_with_dir.find(indented_paragraph) + len(indented_paragraph)
    )
    text_with_dir = text_with_dir[:end_paragraph_idx]
    
    return text_with_dir, indented_paragraph

def rst_math_to_latex_directive(rst_text):
    rst_math = rst_text
    rst_with_html_math = rst_text
    
    while True:        
        rst_math_with_dir, indented_paragraph = rst_extract_directive_block(
            rst_math, '.. math::'
        )
        if not rst_math_with_dir:
            break
        
        # Remove multiple spacing with single one
        clean_paragraph = re.sub(r' +', ' ', indented_paragraph)
        
        # Remove trailing spaces and new lines chars
        clean_paragraph = clean_paragraph.strip('\n')
        clean_paragraph = clean_paragraph.strip()
        
        # Remove spaces after new line char
        clean_paragraph = clean_paragraph.replace('\n ', '\n')
        
        # Replace end of lines with <br>
        html_paragraph = clean_paragraph.replace('\n', '<br>')
        
        # Replace rst dir with html dir
        html_paragraph = f'<latex>{html_paragraph}</latex>'
        
        # Insert html dir
        rst_with_html_math = rst_with_html_math.replace(
            rst_math_with_dir, html_paragraph
        )
        
        # Remove current note block
        end_current_block_idx = (
            rst_math.find(rst_math_with_dir) + len(rst_math_with_dir)
        )
        rst_math = rst_math[end_current_block_idx:]
    
    # Fix not needed multiple new lines
    rst_with_html_math = rst_with_html_math.replace('</latex><br>', '</latex>')
    
    return rst_with_html_math

def repl_upper_html_bold(matched: re.Match):
    return f'<b>{matched.group(1).capitalize()}:</b>'

def rst_urls_to_hrefs_mapper(rst_text):
    labels_urls = re.findall(r'\.\. _([A-Za-z0-9_\- ]+)\: (.+)', rst_text)
    label_to_hrefs_mapper = {}
    for label, url in labels_urls:
        label_to_hrefs_mapper[f'`{label}`_'] = href(label, url)
    return label_to_hrefs_mapper

def rst_code_blocks_to_pre_html(rst_text):
    # Remove the language directive of the code blocks
    rst_text = re.sub(r'\.\. code-block::.*', '.. code-block::', rst_text)
    rst_with_html_pre = rst_text
    
    while True:        
        rst_code_block, indented_paragraph = rst_extract_directive_block(
            rst_text, f'.. code-block::'
        )
        if not rst_code_block:
            break
        
        # Remove multiple spacing with single one
        html_paragraph = re.sub(r' +', ' ', indented_paragraph)
        
        # Remove trailing spaces and new lines chars
        html_paragraph = html_paragraph.strip('\n')
        html_paragraph = html_paragraph.strip()
        
        # Replace new line chars with <br>
        html_paragraph = html_paragraph.replace('\n', '<br>')
        html_paragraph = html_paragraph.replace('<br> ', '<br>')
        
        pre_code_html = f'<pre><code>{html_paragraph}</code></pre>'
        
        # Replace rst note with html note
        rst_with_html_pre = rst_with_html_pre.replace(
            rst_code_block, pre_code_html
        )
        
        # Remove current admon block
        end_current_block_idx = (
            rst_with_html_pre.find(rst_code_block) 
            + len(rst_code_block)
        )
        rst_text = rst_text[end_current_block_idx:]
    return rst_with_html_pre

def rst_to_qt_html(rst_sub_text, rst_global_text=''):
    valid_chars = r'[,A-Za-z0-9μ\-\.=_ \<\>\(\)\\\&;]'
    html_text = rst_sub_text.strip('\n')
    html_text = html_text.replace('<', '&lt;')
    html_text = html_text.replace('>', '&gt;')
    
    label_to_hrefs_mapper = rst_urls_to_hrefs_mapper(rst_global_text)
    
    for label, href_url in label_to_hrefs_mapper.items():
        html_text = html_text.replace(label, href_url)

    html_text = re.sub(
        rf'\.\. confval:: ({valid_chars}+)\n', r'<b>\1:</b>\n', html_text
    )
    for admonition_type in ADMONITION_TYPES:
        html_text = rst_admonitions_to_html_table(html_text, admonition_type)
    
    html_text = rst_code_blocks_to_pre_html(html_text)
    html_text = rst_math_to_latex_directive(html_text)
    html_text = html_text.replace('\n', '<br>')
    html_text = html_text.replace(' <br>', '<br>')
    html_text = html_text.replace('<br> ', '<br>')
    html_text = html_text.replace('<br> ', '<br>')
    html_text = re.sub(rf'``({valid_chars}+)``', r'<code>\1</code>', html_text)
    html_text = re.sub(rf':m:`({valid_chars}+)`', r'<code>\1</code>', html_text)
    html_text = re.sub(
        rf':ref:`({valid_chars}+)`', r'<code>\1</code>', html_text
    )
    html_text = re.sub(
        rf':confval:`({valid_chars}+)`', r'<code>\1</code>', html_text
    )
    html_text = re.sub(rf'\*\*({valid_chars}+)\*\*', r'<b>\1</b>', html_text)
    html_text = re.sub(rf'\*({valid_chars}+)\*', r'<i>\1</i>', html_text)
    html_text = re.sub(rf'`({valid_chars}+)`_', r'<b>\1</b>', html_text)
    
    html_text = re.sub(r':(\w+):', repl_upper_html_bold, html_text)
    
    html_text = html_text.replace('<br><br><latex>', '<br><latex>')
    return html_text

if __name__ == '__main__':
    text = 'ciao'
    print(paragraph(text))
    print(italic(text))
    print(bold(text))
    print(tag(text))
