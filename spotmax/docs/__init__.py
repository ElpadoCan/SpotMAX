import os
import re

from ..html_func import simple_rst_to_html

# Paths
docs_path = os.path.dirname(os.path.abspath(__file__))
source_path = os.path.join(docs_path, 'source')

single_spot_features_filename = 'single_spot_features_description.rst'
single_spot_features_rst_filepath = os.path.join(
    source_path, single_spot_features_filename
)

aggr_features_filename = 'aggr_features_description.rst'
aggr_features_rst_filepath = os.path.join(source_path, aggr_features_filename)

# Urls
readthedocs_url = 'spotmax.rtfd.io/en/latest'

single_spot_features_desc_url = (
    f'{readthedocs_url}/{single_spot_features_filename}.html'
)
aggr_features_desc_url = (
    f'{readthedocs_url}/{aggr_features_filename}.html'
)

# Regex patterns
metric_name_regex = r'[A-Za-z0-9_ \-\.\(\)`\^]+'
col_name_regex = r'[A-Za-z0-9_]+'

def _get_section(idx, groups, rst_text):
    rst_text = re.sub(r'\.\. (.*)\n', '', rst_text)
    group = groups[idx]
    start_idx = rst_text.find(group)
    if (idx+1) == len(groups):
        section = rst_text[start_idx:]
    else:
        stop_idx = rst_text.find(groups[idx+1])
        section = rst_text[start_idx:stop_idx]
    return section

def single_spot_feature_group_name_to_url(group_name):
    url_tag = re.sub(r'[^a-zA-Z0-9]+', '-', group_name.lower()).strip('-')
    infoUrl = f'{single_spot_features_desc_url}#{url_tag}'
    return infoUrl

def aggr_feature_group_name_to_url(group_name):
    url_tag = re.sub(r'[^a-zA-Z0-9]+', '-', group_name.lower()).strip('-')
    infoUrl = f'{aggr_features_desc_url}#{url_tag}'
    return infoUrl

def _parse_features_groups(rst_text):
    features_groups = {}
    group_pattern = r'\n(.+)\n\-+\n'
    feature_pattern = r'\* \*\*(.+)\*\*:'
    groups = re.findall(group_pattern, rst_text)
    
    for g, group in enumerate(groups):
        section = _get_section(g, groups, rst_text)        
        features_names = re.findall(feature_pattern, section)
        features_groups[group] = features_names
    return features_groups

def parse_single_spot_features_groups():
    with open(single_spot_features_rst_filepath, 'r') as rst:
        rst_text = rst.read()
    
    features_groups = _parse_features_groups(rst_text)
        
    return features_groups

def parse_aggr_features_groups():
    with open(aggr_features_rst_filepath, 'r') as rst:
        rst_text = rst.read()
    
    features_groups = _parse_features_groups(rst_text)
        
    return features_groups

def _parse_column_names(features_groups, rst_text):
    mapper = {}
    groups = list(features_groups.keys())
    for g, group in enumerate(groups):
        section = _get_section(g, groups, rst_text)
        for metric_name in features_groups[group]:
            escaped = re.escape(metric_name)
            pattern =(
                fr'\* \*\*{escaped}\*\*: column name ``({col_name_regex})``'
            )

            column_name = re.findall(pattern, section)[0]
                
            key = f'{group}, {metric_name}'
            mapper[key] = column_name
    return mapper

def parse_aggr_features_column_names():
    with open(aggr_features_rst_filepath, 'r') as rst:
        rst_text = rst.read()
        
    features_groups = parse_aggr_features_groups()
    mapper = _parse_column_names(features_groups, rst_text)
    return mapper
        