import sys
import os
import re
import tempfile

# Inspired from https://stackoverflow.com/questions/65502005/convert-a-jupyter-notebook-to-html-output-in-native-python
# See manual here : https://buildmedia.readthedocs.org/media/pdf/nbconvert/latest/nbconvert.pdf
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import ASCIIDocExporter
from nbconvert import HTMLExporter
from nbconvert import PDFExporter


# This script:
# - executes a jupyter notebook test script (argv[1])
# - converts it into a file saved in the given directory (argv[2]).
# - according to the format [asciidoc, html, pdf] (argv[3])
python_exe = os.path.realpath(sys.executable)
test_script = sys.argv[1]
out_dir = sys.argv[2]
test_name = os.path.splitext(os.path.basename(test_script))[0]  # No extension
out_type = "asciidoc"
if len(sys.argv) > 3:
    out_type = sys.argv[3]
test_output = os.path.join(out_dir, test_name + "." + out_type)

# Read [and hack source notebook in a temporary notebook]
f = open(test_script, "r", encoding="utf8")
if out_type == "asciidoc":
    # Kill some cells that pollute nonregression
    nbs = f.read()
    nbs = re.sub(r"from IPython.display import Markdown", "", nbs)
    nbs = re.sub(r"Markdown", "print", nbs)
    print(type(nbs))
    new_file, filename = tempfile.mkstemp(text=True)
    os.write(new_file, nbs.encode("utf8"))
    os.close(new_file)
    f = open(filename, "r", encoding="utf8")

# Really read the notebook
nb = nbformat.read(f, as_version=nbformat.NO_CONVERT)

# Execute the Notebook
ep = ExecutePreprocessor(timeout=-1, kernel_name="python3")
ep.preprocess(nb)

# Export to asciidoc (dump only output cells for test purpose)
if out_type == "asciidoc":
    exporter = ASCIIDocExporter()
    exporter.exclude_input = True
    exporter.exclude_input_prompt = True
    exporter.exclude_markdown = True
    exporter.exclude_raw = True
    exporter.exclude_unknown = True
# Export to HTML
elif out_type == "html":
    exporter = HTMLExporter()
    # Ensure that equations and 3D is well displayed!
    exporter.mathjax_url = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js?config=TeX-MML-AM_CHTML"
    exporter.require_js_url = (
        "https://requirejs.org/docs/release/2.3.6/minified/require.js"
    )
# Export to PDF
elif out_type == "pdf":
    exporter = PDFExporter()
else:
    print("Wrong output file type for run_test_ipynb.py [asciidoc, html, pdf]")

# Export the notebook
notebook_node, resources = exporter.from_notebook_node(nb)

# Write to output file
if out_type == "pdf":
    with open(test_output, "wb") as f:
        f.write(notebook_node)
else:
    with open(test_output, "w", encoding="utf8") as f:
        f.write(notebook_node)
