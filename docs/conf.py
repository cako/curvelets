from __future__ import annotations

import importlib.metadata

project = "Curvelets"
copyright = "2024, Carlos Alberto da Costa Filho"
author = "Carlos Alberto da Costa Filho"
version = release = importlib.metadata.version("curvelets")

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_gallery.gen_gallery",
    "sphinxcontrib.bibtex",
]

source_suffix = [".rst", ".md"]
exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
    ".env",
    ".venv",
]

html_theme = "furo"

myst_enable_extensions = [
    "colon_fence",
]

nitpick_ignore = [
    ("py:class", "_io.StringIO"),
    ("py:class", "_io.BytesIO"),
    ("py:class", "C"),
    ("py:class", "F"),
    ("py:class", "T"),
    ("py:class", "U"),
    ("py:class", "optional"),
    ("py:class", '"curvelet"'),
    ("py:class", '"meyer"'),
    ("py:class", '"wavelet"'),
    ("py:class", '{"curvelet"'),
    ("py:class", '"wavelet"}'),
    ("py:class", "ParamUDCT"),
    ("py:class", "UDCTWindows"),
]

always_document_param_types = True

# sphinx.ext.intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# sphinx_copybutton
copybutton_prompt_text = ">>> "

# sphinx_gallery.gen_gallery
sphinx_gallery_conf = {
    "examples_dirs": "../examples",  # path to your example scripts
    "gallery_dirs": "auto_examples",  # path to where to save gallery generated output
    "within_subsection_order": "FileNameSortKey",
}

# sphinxcontrib.bibtex
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"
# bibtex_default_style = "plain"
suppress_warnings = ["bibtex.duplicate_citation"]
