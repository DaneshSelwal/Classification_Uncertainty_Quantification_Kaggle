import nbformat

nb = nbformat.read('Classification/baseline/Model_training.ipynb', as_version=4)

# Verify Kaggle environment guards and paths
for cell in nb.cells:
    if cell.cell_type == 'code':
        if 'drive.mount' in cell.source and 'IN_COLAB' not in cell.source:
            cell.source = "import os\nIN_KAGGLE = True\n"
        if '"/kaggle/input/classification-uq-data"' in cell.source:
            cell.source = cell.source.replace('"/kaggle/input/classification-uq-data"', '"/kaggle/input/dias-6-bands-data"')

nbformat.write(nb, 'Classification/baseline/Model_training.ipynb')
print("Verified notebook configuration.")
