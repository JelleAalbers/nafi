

# Trick stolen from iminuit: make first page show README.md content,
# with a table of contents added

with open("../README.md") as f:
    readme_content = f.read()
readme_content = readme_content.replace("docs/", "")

with open("index.md.in") as f:
    index_content = f.read()

with open("index.md", "w") as f:
    f.write(readme_content + index_content)


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'nafi'
copyright = '2023, Jelle Aalbers'
author = 'Jelle Aalbers'
release = '0.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']


# From https://stackoverflow.com/a/43186995: add custom css to widen the page
def setup(app):
    app.add_css_file('make_it_wider.css')

