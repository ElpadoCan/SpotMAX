import os
import re

from urllib.parse import urlparse

from spotmax import html_func

# Paths
docs_path = os.path.dirname(os.path.abspath(__file__))
source_path = os.path.join(docs_path, 'source')

single_spot_features_filename = 'single_spot_features_description.rst'
single_spot_features_rst_filepath = os.path.join(
    source_path, single_spot_features_filename
)

aggr_features_filename = 'aggr_features_description.rst'
aggr_features_rst_filepath = os.path.join(source_path, aggr_features_filename)

params_desc_filename = 'parameters_description.rst'
params_desc_rst_filepath = os.path.join(source_path, params_desc_filename)

# Urls
readthedocs_url = 'https://spotmax.readthedocs.io/'

single_spot_features_desc_url = (
    f'{readthedocs_url}/{single_spot_features_filename}.html'
)
aggr_features_desc_url = (
    f'{readthedocs_url}/{aggr_features_filename}.html'
)
params_desc_desc_url = (
    f'{readthedocs_url}/{params_desc_filename}.html'
)

# Regex patterns
metric_name_regex = r'[A-Za-z0-9_ \-\.\(\)`\^]+'
col_name_regex = r'[A-Za-z0-9_]+'
option_pattern = r'\* \*\*(.+)\*\*:'

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

def params_desc_section_to_url(section):
    url_tag = re.sub(r'[^a-zA-Z0-9]+', '-', section.lower()).strip('-')
    infoUrl = f'{params_desc_desc_url}#{url_tag}'
    return infoUrl

def get_params_desc_mapper():
    with open(params_desc_rst_filepath, 'r') as rst:
        rst_text = rst.read()
    
    section_options_mapper = _parse_section_options(rst_text)
    section_option_desc_mapper = _parse_desc(rst_text, section_options_mapper)
    return section_option_desc_mapper

def _parse_section_options(rst_text):
    features_groups = {}
    group_pattern = r'\n(.+)\n\-+\n'
    groups = re.findall(group_pattern, rst_text)
    
    for g, group in enumerate(groups):
        section = _get_section(g, groups, rst_text)        
        features_names = re.findall(option_pattern, section)
        features_groups[group] = features_names
    return features_groups

def _underline_header(text, underline_char='-'):
    underline = f'{underline_char}'*len(text)
    underlined = f'{text}\n{underline}'
    return underlined

def _parse_desc(rst_text, section_options_mapper, to_html=True):
    s = 0
    section_option_mapper = {}
    sections = list(section_options_mapper.keys())
    for section, options in section_options_mapper.items():
        header = _underline_header(section)
        if s+1 < len(sections)-1:
            next_section = sections[s+1]
            next_header = _underline_header(next_section)
            stop_idx = rst_text.find(next_header)
        else:
            stop_idx = -1
        start_idx = rst_text.find(header) + len(header)
        section_text = rst_text[start_idx:stop_idx]
        n = 0
        num_options = len(options)
        for option in options:
            if n+1 < num_options-1:
                next_option = options[n+1]
                next_nth_option_txt = f'* **{next_option}**:'
                option_stop_idx = section_text.find(next_nth_option_txt)
            else:
                option_stop_idx = -1
            nth_option_txt = f'* **{option}**:'
            option_start_idx = section_text.find(nth_option_txt) + 2
            desc = section_text[option_start_idx:option_stop_idx]
            if to_html:
                desc = html_func.simple_rst_to_html(desc)
            section_option_mapper[(section, option)] = desc
            n += 1
        s += 1

    return section_option_mapper

def parse_single_spot_features_groups():
    with open(single_spot_features_rst_filepath, 'r') as rst:
        rst_text = rst.read()
    
    features_groups = _parse_section_options(rst_text)
        
    return features_groups

def parse_aggr_features_groups():
    with open(aggr_features_rst_filepath, 'r') as rst:
        rst_text = rst.read()
    
    features_groups = _parse_section_options(rst_text)
        
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

            try:
                column_name = re.findall(pattern, section)[0]
            except Exception as e:
                import pdb; pdb.set_trace()
                
            key = f'{group}, {metric_name}'
            mapper[key] = column_name
    return mapper

def parse_aggr_features_column_names():
    with open(aggr_features_rst_filepath, 'r') as rst:
        rst_text = rst.read()
        
    features_groups = parse_aggr_features_groups()
    mapper = _parse_column_names(features_groups, rst_text)
    return mapper

def single_spot_features_column_names():
    with open(single_spot_features_rst_filepath, 'r') as rst:
        rst_text = rst.read()
        
    features_groups = parse_single_spot_features_groups()
    mapper = _parse_column_names(features_groups, rst_text)
    return mapper