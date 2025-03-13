# Configuration file for the Sphinx documentation builder.

from datetime import datetime
import spotmax

# -- Report bug information
import subprocess
command = 'sphinx-build --bug-report'
try:
    subprocess.check_call([command], shell=True)
except Exception as err:
    subprocess.check_call(command.split(), shell=True)

# -- Project information

project = 'SpotMAX'
author = spotmax.__author__
copyright = f'{datetime.now():%Y}, {author}'

version = spotmax.__version__
release = version


# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    # 'sphinx.ext.autosectionlabel',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx_copybutton',
    'sphinxcontrib.email',
    'sphinx_tabs.tabs',
    'sphinx_toolbox.confval',
    # 'sphinx_carousel.carousel',
    'sphinxcontrib.video',
    # # 'nbsphinx'
    # # 'sphinx_rtd_dark_mode'
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for EPUB output
epub_show_urls = 'footnote'

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'
# html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_favicon = "_static/favicon.ico"
html_logo = "_static/logo.png"

html_context = {
    'display_github': True, # Integrate GitHub
    'github_user': 'ElpadoCan', 
    'github_repo': 'SpotMAX',
    'github_version': 'main',
    'conf_py_path': '/spotmax/docs',
}

# -- My css
html_css_files = [
    # 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css'
    'css/custom.css',
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'default'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = [
    '_build', 
    'Thumbs.db', 
    '.DS_Store', 
    'source/features/_background_description.rst', 
    'source/features/_effect_size_description.rst',
    'source/features/_effect_size_Cohen_formula.rst',
    'source/features/_effect_size_Glass_formula.rst',
    'source/features/_effect_size_Hedge_formula.rst',
    'source/install/_install_numba.rst',
    'source/install/_install_conda_open_terminal.rst',
    'source/install/_conda_create_activate_acdc.rst',
    'source/install/_gui_packages.rst'
]

# Set html options for the theme
html_theme_options = {
    # 'includehidden': True,
    'navigation_depth': 3,
}

language = 'en'