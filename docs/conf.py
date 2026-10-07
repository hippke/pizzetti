import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "pizzetti"
author = "Author One"
copyright = "2026, Author One"
try:
    from pizzetti import __version__ as release
except Exception:  # pragma: no cover
    release = "0.1.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence"]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
html_theme = "furo"
html_title = "pizzetti"
autodoc_member_order = "bysource"
exclude_patterns = ["_build"]
