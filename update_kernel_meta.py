import json
from pathlib import Path

# Sync baseline metadata with active 6-band Kaggle settings
meta_path = Path('Classification/baseline/kernel-metadata.json')
if meta_path.exists():
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    meta['id'] = 'daneshselwal/cuq6-baseline-k01'
    meta['title'] = 'cuq6-baseline-k01'
    meta['code_file'] = 'Model_training.ipynb'
    meta['language'] = 'python'
    meta['kernel_type'] = 'notebook'
    meta['enable_gpu'] = True
    meta['dataset_sources'] = ['daneshselwal/dias-6-bands-data']

    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print("Updated Classification/baseline/kernel-metadata.json")
