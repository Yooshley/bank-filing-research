"""Reuse notebook definitions without executing installs, examples or API calls."""
import ast
import json
from pathlib import Path


def load_definitions(path, namespace, assignments=()):
    notebook = json.loads(Path(path).read_text(encoding='utf-8'))
    statements = []
    for cell in notebook['cells']:
        source = ''.join(cell.get('source', []))
        if cell['cell_type'] != 'code' or source.lstrip().startswith('%'):
            continue
        for node in ast.parse(source).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                statements.append(node)
            elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in assignments for t in node.targets):
                statements.append(node)
    exec(compile(ast.Module(body=statements, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace
