import os
import re
import traceback
from pathlib import Path

from urllib.parse import urlparse

from .. import html_func, printl

# Paths
docs_path = os.path.dirname(os.path.abspath(__file__))
source_path = os.path.join(docs_path, 'source')
docs_html_path = os.path.join(docs_path, '_build', 'html')

single_spot_features_filename = 'single_spot_features_description'
single_spot_features_relpath = f'features/{single_spot_features_filename}'
single_spot_features_rst_filepath = os.path.join(
    source_path, *single_spot_features_relpath.split('/')
)
single_spot_features_rst_filepath = f'{single_spot_features_rst_filepath}.rst'

aggr_features_filename = 'aggr_features_description'
aggr_features_relpath = f'features/{aggr_features_filename}'
aggr_features_rst_filepath = os.path.join(
    source_path, *aggr_features_relpath.split('/')
)
aggr_features_rst_filepath = f'{aggr_features_rst_filepath}.rst'

ref_ch_features_filename = 'ref_ch_features_description'
ref_ch_features_relpath = f'features/{ref_ch_features_filename}'
ref_ch_features_rst_filepath = os.path.join(
    source_path, *ref_ch_features_relpath.split('/')
)
ref_ch_features_rst_filepath = f'{ref_ch_features_rst_filepath}.rst'

params_desc_filename = 'parameters_description'
params_desc_relpath = f'parameters/{params_desc_filename}'
params_desc_rst_filepath = os.path.join(
    source_path, *params_desc_relpath.split('/')
)
params_desc_rst_filepath = f'{params_desc_rst_filepath}.rst'

# Urls
readthedocs_url = 'https://spotmax.readthedocs.io/en/latest'
single_spot_features_desc_url = (
    f'{readthedocs_url}/{single_spot_features_relpath}.html'
)
aggr_features_desc_url = (
    f'{readthedocs_url}/{aggr_features_relpath}.html'
)
params_desc_desc_url = (
    f'{readthedocs_url}/{params_desc_relpath}.html'
)

# Local html filepaths
single_spot_features_desc_html_filepath = os.path.join(
    docs_html_path, *single_spot_features_relpath.split('/')
)
single_spot_features_desc_html_filepath = (
    f'{single_spot_features_desc_html_filepath}.html'
)

aggr_features_desc_html_filepath = os.path.join(
    docs_html_path, *aggr_features_relpath.split('/')
)
aggr_features_desc_html_filepath = f'{aggr_features_desc_html_filepath}.html'

params_desc_desc_html_filepath = os.path.join(
    docs_html_path, *params_desc_relpath.split('/')
)
params_desc_desc_html_filepath = f'{params_desc_desc_html_filepath}.html'

# Regex patterns
metric_name_regex = r'[A-Za-z0-9_ \-\.\(\)`\^]+'
col_name_regex = r'[A-Za-z0-9_]+'
confval_pattern = r'\.\. confval:: (.+)\n'
ul_item_pattern = r'\* \*\*(.+)\*\*:'

def read_rst(rst_filepath):
    with open(rst_filepath, 'r', encoding='utf-8') as rst:
        rst_text = rst.read()
    
    rst_folderpath = os.path.dirname(rst_filepath)
    includes = re.findall(r'.. include:: ([A-Za-z0-9\._-]+)', rst_text)
    for include_filename in includes:
        abspath = (Path(rst_folderpath) / include_filename).resolve()
        with open(abspath, 'r', encoding='utf-8') as txt:
            include_text = re.escape(txt.read())
        try:
            rst_text = re.sub(
                rf'.. include:: {include_filename}', include_text, rst_text
            )
            rst_text = rst_text.replace('\\', '')
        except Exception as err:
            traceback.print_exc()
            import pdb; pdb.set_trace()
    return rst_text 
    
def _get_section(idx, groups, rst_text, remove_directives=True):
    if remove_directives:
        rst_text = re.sub(r'\.\. (.*)\n', '', rst_text)
    
    header = _underline_header(groups[idx])
    start_idx = rst_text.find(header)
    if (idx+1) == len(groups):
        section = rst_text[start_idx:]
    else:
        next_header = _underline_header(groups[idx+1])
        stop_idx = rst_text.find(next_header)
        section = rst_text[start_idx:stop_idx]

    return section

def norm_url_tag(text):
    return re.sub(r'[^a-zA-Z0-9]+', '-', text).strip('-')

def single_spot_feature_group_name_to_url(group_name):
    url_tag = norm_url_tag(group_name)
    infoUrl = f'{single_spot_features_desc_url}#{url_tag}'
    return infoUrl

def aggr_feature_group_name_to_url(group_name):
    url_tag = norm_url_tag(group_name)
    infoUrl = f'{aggr_features_desc_url}#{url_tag}'
    return infoUrl

def params_desc_section_to_url(section):
    url_tag = norm_url_tag(section)
    infoUrl = f'{params_desc_desc_url}#{url_tag}'
    return infoUrl

def param_name_to_url(param_name):
    param_tag = norm_url_tag(param_name)
    confval_tag = f'#confval-{param_tag}'
    url = f'{params_desc_desc_url}{confval_tag}'
    return url

def param_name_to_local_html_url(param_name):
    param_tag = norm_url_tag(param_name)
    confval_tag = f'#confval-{param_tag}'
    url = f'file://{params_desc_desc_html_filepath}{confval_tag}'
    return url

def get_params_desc_mapper():
    rst_text = read_rst(params_desc_rst_filepath)
    
    section_options_mapper = _parse_section_options(
        rst_text, confval_pattern, return_sections=True
    )    
    section_option_desc_mapper = _parse_desc(section_options_mapper, rst_text)
    return section_option_desc_mapper

def _parse_section_options(rst_text, option_pattern, return_sections=False):
    features_groups = {}
    group_pattern = r'\n(.+)\n\-+\n'
    groups = re.findall(group_pattern, rst_text)
    
    for g, group in enumerate(groups):
        section = _get_section(g, groups, rst_text, remove_directives=False)        
        features_names = re.findall(option_pattern, section)
        if return_sections:
            features_groups[group] = features_names, section
        else:
            features_groups[group] = features_names
    return features_groups

def _underline_header(text, underline_char='-'):
    underline = f'{underline_char}'*len(text)
    underlined = f'{text}\n{underline}'
    return underlined

def _parse_desc(section_options_mapper, rst_text, to_html=True):
    section_option_mapper = {}
    for group, (options, section_text) in section_options_mapper.items():
        num_options = len(options)
        for n, option in enumerate(options):
            if n+1 < num_options:
                next_option = options[n+1]
                next_nth_option_txt = f'.. confval:: {next_option}\n'
                option_stop_idx = section_text.find(next_nth_option_txt)
            else:
                option_stop_idx = -1
            nth_option_txt = f'.. confval:: {option}\n'
            option_start_idx = section_text.find(nth_option_txt)
            desc = section_text[option_start_idx:option_stop_idx]
            # Remove rst labels for cross-ref
            desc = re.sub(r'\.\. _(.*)\n', '', desc)
            if to_html:
                desc = html_func.rst_to_qt_html(desc, rst_global_text=rst_text)
            section_option_mapper[(group, option)] = desc
    return section_option_mapper

def parse_single_spot_features_groups():
    rst_text = read_rst(single_spot_features_rst_filepath)
    
    features_groups = _parse_section_options(rst_text, ul_item_pattern)
        
    return features_groups

def parse_ref_ch_featurs_groups():
    rst_text = read_rst(ref_ch_features_rst_filepath)
    
    features_groups = _parse_section_options(rst_text, ul_item_pattern)
        
    return features_groups

def parse_aggr_features_groups():
    rst_text = read_rst(aggr_features_rst_filepath)
    
    features_groups = _parse_section_options(rst_text, ul_item_pattern)
        
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
            except Exception as err:
                traceback.print_exc()
                import pdb; pdb.set_trace()
                
            key = f'{group}, {metric_name}'
            mapper[key] = column_name
            
    return mapper

def parse_aggr_features_column_names():
    rst_text = read_rst(aggr_features_rst_filepath)
        
    features_groups = parse_aggr_features_groups()
    mapper = _parse_column_names(features_groups, rst_text)
    return mapper

def single_spot_features_column_names():
    rst_text = read_rst(single_spot_features_rst_filepath)
        
    features_groups = parse_single_spot_features_groups()
    mapper = _parse_column_names(features_groups, rst_text)

    return mapper

def ref_ch_features_column_names():
    rst_text = read_rst(ref_ch_features_rst_filepath)
        
    features_groups = parse_ref_ch_featurs_groups()
    mapper = _parse_column_names(features_groups, rst_text)
    return mapper