# Configuration file for the Sphinx documentation builder.

from datetime import datetime
import spotmax

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
    # 'sphinx_rtd_dark_mode'
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

# -- My css
html_css_files = [
    'css/custom.css',
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Set html options for the theme
html_theme_options = {
    'includehidden': True,
}

language = 'en'