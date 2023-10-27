import os
import re

from ..html_func import simple_rst_to_html

# Paths
docs_path = os.path.dirname(os.path.abspath(__file__))
source_path = os.path.join(docs_path, 'source')
features_rst_filepath = os.path.join(source_path, 'features_description.rst')

# Urls
readthedocs_url = 'spotmax.rtfd.io/en/latest'
features_desc_url = f'{readthedocs_url}/features_description.html'

def _get_section(idx, groups, rst_text):
    group = groups[idx]
    start_idx = rst_text.find(group)
    if (idx+1) == len(groups):
        section = rst_text[start_idx:]
    else:
        stop_idx = rst_text.find(groups[idx+1])
        section = rst_text[start_idx:stop_idx]
    return section

def feature_group_name_to_url(group_name):
    url_tag = re.sub(r'[^a-zA-Z0-9]+', '-', group_name.lower()).strip('-')
    infoUrl = f'{features_desc_url}#{url_tag}'
    return infoUrl

def parse_features_groups():
    features_groups = {}
    with open(features_rst_filepath, 'r') as rst:
        rst_text = rst.read()
    
    group_pattern = r'\n(.+)\n\-+\n'
    feature_pattern = r'\* \*\*(.+)\*\*:'
    groups = re.findall(group_pattern, rst_text)
    
    for g, group in enumerate(groups):
        section = _get_section(g, groups, rst_text)        
        features_names = re.findall(feature_pattern, section)
        features_groups[group] = features_names
        
    return features_groups

def parse_features_description():
    with open(features_rst_filepath, 'r') as rst:
        rst_text = rst.read()
    
    column_name_pattern = r'\* \*\*<name>\*\*: column name ``([a-zA-Z0-9]+)``\.'
    features_groups = parse_features_groups()
    groups = list(features_groups.keys())
    for g, group in enumerate(groups):
        feature_names = features_groups[group]
        section = _get_section(g, groups, rst_text)
        first_feature_text = f'* **{feature_names[0]}**:'
        stop_idx = section.find(first_feature_text)
        underline_text = '-'*len(group)
        start_idx = section.find(underline_text)
        start_idx += len(underline_text)
        general_desc = section[start_idx:stop_idx]
        desc_html = simple_rst_to_html(general_desc)
        
        from cellacdc.widgets import myMessageBox
        from qtpy.QtWidgets import QApplication
        app = QApplication([])
        msg = myMessageBox(wrapText=False)
        msg.information(None, 'Test', desc_html)
        import pdb; pdb.set_trace()
        for feature_name in feature_names:
            pattern = column_name_pattern.replace('<name>', feature_name)
            column_name = re.findall(fr'{pattern}', section)[0]
            import pdb; pdb.set_trace()