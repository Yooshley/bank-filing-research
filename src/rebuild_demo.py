"""Rebuild local demo data using notebooks 1 and 2 without paid embeddings."""
import os
import re
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    base = Path(__file__).resolve().parents[1]
    output = base / '.local' / 'rebuild'
    output.mkdir(parents=True, exist_ok=True)
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    for name in ('01_data_ingestion.ipynb', '02_embedding_vectordb.ipynb'):
        notebook = nbformat.read(base / 'notebooks' / name, as_version=4)
        for cell in notebook.cells:
            if cell.cell_type != 'code':
                continue
            if cell.source.lstrip().startswith(('%pip', '!pip')):
                cell.source = '# Dependencies already installed.'
            cell.source = re.sub(r'^USE_OPENAI\s*=.*$', 'USE_OPENAI = False', cell.source, flags=re.M)
        notebook.cells.insert(0, nbformat.v4.new_code_cell(
            'import os\nos.environ["OMP_NUM_THREADS"] = "2"\n'
            'import torch\ntorch.set_num_threads(2)'))
        print(f'Building {name}', flush=True)
        def progress(cell, cell_index, **kwargs):
            if cell.cell_type == 'code':
                print(f'{name}: cell {cell_index}', flush=True)
        client = NotebookClient(notebook, timeout=1800, kernel_name='python3',
                                resources={'metadata': {'path': str(base)}},
                                on_cell_start=progress)
        try:
            client.execute()
        finally:
            nbformat.write(notebook, output / name)
        print(f'Completed {name}', flush=True)


if __name__ == '__main__':
    main()
