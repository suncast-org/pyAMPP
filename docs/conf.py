# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config


# -- Project information -----------------------------------------------------

# The full version, including alpha/beta/rc tags
release = "0.1.1.1"

project = "pyAMPP"
copyright = "2022, suncast-org"
author = "suncast-org"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',  # Automatically document your code
    'sphinx.ext.napoleon',  # Support for Google-style docstrings
    'sphinx.ext.viewcode',  # Add links to source code from documentation
    'sphinx.ext.mathjax',  # Render math via JavaScript
    'sphinx.ext.autosummary',  # Automatically generates summary tables from the docstrings.
    'sphinx.ext.githubpages',
    'sphinx.ext.graphviz',
    'sphinx.ext.imgmath',
    # 'sphinx_gallery.gen_gallery',
    # 'sphinx_gallery',
    'autoapi.extension'
    # Add any other Sphinx extensions here.
]

# Add any paths that contain templates here, relative to this directory.
# templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'

html_static_path = ['_static']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {"python": ("https://docs.python.org/", None)}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "alabaster"


# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using a default sidebar with
# the links to the documentation's roots, contents & search, plus a link to
# the Python.org website.
html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        'relations.html',  # needs 'show_related': True theme option to display
        'searchbox.html',
        'donate.html',
    ]
}

# -- Options for sphinx.ext.autodoc -------------------------------------------
autoapi_type = 'python'
autoapi_dirs = ['../pyampp']
autoapi_ignore = [
    'tests/*',
    '_dev/*',
    'lib/*'
]
