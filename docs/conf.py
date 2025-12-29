from __future__ import annotations

import sys
from pathlib import Path

# Add the src directory to the path to import the version directly from source
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from curvelets._version import version as _version

project = "Curvelets"
copyright = "2026, Carlos Alberto da Costa Filho"
author = "Carlos Alberto da Costa Filho"
# Version is read directly from source. When building from a tag, hatch-vcs
# generates a clean version (e.g., "0.1.0b2"). When building from a commit
# after a tag, it includes dev metadata (e.g., "0.1.0b2.dev1+...").
# For stable docs, Read the Docs should build from release tags.
version = release = _version

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
    ("py:class", '"wavelet"'),
    ("py:class", '{"curvelet"'),
    ("py:class", '"wavelet"}'),
    ("py:class", "ParamUDCT"),
    ("py:class", "UDCTWindows"),
    ("py:class", "MUDCTCoefficients"),
    ("py:obj", "MUDCTCoefficients"),
    ("py:obj", "curvelets.numpy.MUDCTCoefficients"),
    ("py:class", "UDCTCoefficients"),
    ("py:obj", "UDCTCoefficients"),
    ("py:class", '{"real"'),
    ("py:class", '"complex"'),
    ("py:class", '"monogenic"}'),
]

always_document_param_types = True

# sphinx.ext.intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
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
