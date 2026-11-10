"""Portable cloud index integrity and restoration checks without network calls."""
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import chromadb
from src.cloud_index import ensure_cloud_index, export_bundle


class CloudIndexTests(unittest.TestCase):
    def test_export_restore_and_checksum_guard(self):
        # Chroma can retain SQLite handles until process exit on Windows.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            root = Path(folder)
            config = {'default_provider': 'local', 'models': {'local': 'test-model'},
                      'collections': {'local': 'test_collection'}, 'local_window_tokens': 220}
            for name in ('source', 'target'):
                base = root / name
                (base / 'reports/embedding_vectordb').mkdir(parents=True)
                (base / 'reports/embedding_vectordb/retrieval_config.json').write_text(json.dumps(config))
            collection = chromadb.PersistentClient(path=str(root/'source/models/chroma')).create_collection(
                'test_collection', embedding_function=None)
            collection.add(ids=['BMO_2025_1'], documents=['Public credit risk disclosure'],
                           metadatas=[{'bank_code': 'BMO', 'fiscal_year': 2025}], embeddings=[[1., 0., 0.]])
            archive = root / 'index.zip'
            export_bundle(root/'source', archive)
            raw = archive.read_bytes()
            class Response:
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def raise_for_status(self): pass
                def iter_content(self, size): yield raw
            env = {'INDEX_BUNDLE_URL': 'https://example.com/index.zip',
                   'INDEX_BUNDLE_SHA256': hashlib.sha256(raw).hexdigest()}
            with patch.dict(os.environ, env), patch('requests.get', return_value=Response()) as get:
                ensure_cloud_index(root/'target')
                ensure_cloud_index(root/'target')
                self.assertEqual(get.call_count, 1)
            restored = chromadb.PersistentClient(path=str(root/'target/models/chroma')).get_collection('test_collection')
            self.assertEqual(restored.count(), 1)
            with patch.dict(os.environ, {**env, 'INDEX_BUNDLE_SHA256': '0'*64}), patch('requests.get', return_value=Response()):
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    ensure_cloud_index(root/'target')
            self.assertEqual(restored.count(), 1)
