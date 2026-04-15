import nbformat

file_path = "Seq2Seq_translator_model.ipynb"

nb = nbformat.read(file_path, as_version=4)

if "widgets" in nb.get("metadata", {}):
    del nb["metadata"]["widgets"]

# overwrite the same file (no new notebook created)
nbformat.write(nb, file_path)

print("Fixed Notebook by removing 'widgets' from metadata.")