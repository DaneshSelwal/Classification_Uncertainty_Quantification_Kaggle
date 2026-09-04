import nbformat

nb = nbformat.read('Classification/baseline/Model_training.ipynb', as_version=4)

# find and modify cells
for cell in nb.cells:
    if cell.cell_type == 'code':
        # strip drive mount
        if 'drive.mount' in cell.source:
            cell.source = "import os\nIN_KAGGLE = True\n"
        
        # update paths
        if 'MODEL_DIR   = Path("/content/drive/My Drive/' in cell.source:
            cell.source = cell.source.replace('"/content/drive/My Drive/Classification/baseline/models"', '"/kaggle/working/Classification/baseline/models"')
            cell.source = cell.source.replace('"/content/drive/My Drive/Classification/baseline/results"', '"/kaggle/working/Classification/baseline/results"')
            cell.source = cell.source.replace('"/content/drive/My Drive/Classification/data"', '"/kaggle/input/classification-uq-data"')

nbformat.write(nb, 'Classification/baseline/Model_training.ipynb')
print("Updated notebook.")
