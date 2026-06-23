import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os

print("Reading notebook: fashion_recommendation_assistant.ipynb...")
with open('fashion_recommendation_assistant.ipynb', 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

print("Executing notebook (running all cells)...")
# Run with a 10-minute timeout
ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
ep.preprocess(nb, {'metadata': {'path': os.getcwd()}})

print("Saving executed notebook...")
with open('fashion_recommendation_assistant.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Notebook executed and saved successfully with all cell outputs populated!")
