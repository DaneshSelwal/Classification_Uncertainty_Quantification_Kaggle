import json

with open('Classification/baseline/kernel-metadata.json', 'r') as f:
    meta = json.load(f)

meta['id'] = 'daneshselwal/classification-uq-baseline'
meta['title'] = 'Classification UQ Baseline'
meta['code_file'] = 'Model_training.ipynb'
meta['language'] = 'python'
meta['kernel_type'] = 'notebook'
meta['enable_gpu'] = 'true'
meta['dataset_sources'] = ['daneshselwal/classification-uq-data']

with open('Classification/baseline/kernel-metadata.json', 'w') as f:
    json.dump(meta, f, indent=2)

print("Updated kernel-metadata.json")
