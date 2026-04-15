import nbformat

file_path = "Seq2Seq_translator_model.ipynb"  # change this

nb = nbformat.read(file_path, as_version=4)

if "widgets" in nb.get("metadata", {}):
    del nb["metadata"]["widgets"]

nbformat.write(nb, "cleaned_notebook.ipynb")
print("Done!")