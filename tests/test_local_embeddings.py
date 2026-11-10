"""Offline regressions for the current notebook embedding implementation."""
import unittest
from pathlib import Path
import numpy as np
from src.notebook_utils import load_definitions

ROOT = Path(__file__).resolve().parents[1]

class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))
    def decode(self, tokens):
        return ' '.join(str(token) for token in tokens)

class Model:
    """Record windows and return deterministic normalized test embeddings."""
    tokenizer = Tokenizer()
    max_seq_length = 256
    def __init__(self):
        self.windows = []
    def get_embedding_dimension(self):
        return 3
    def encode(self, windows, **kwargs):
        self.windows = windows
        vectors = np.array([[len(window.split()), 1, 2] for window in windows], dtype=float)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

class LocalEmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.model = Model()
        self.ns = {'np': np, 'local_model': self.model, 'LOCAL_WINDOW_TOKENS': 220,
                   'TOP_K': 5, 'FETCH_K': 30, 'MMR_LAMBDA': .7}
        load_definitions(ROOT / 'notebooks' / '02_embedding_vectordb.ipynb', self.ns)

    def test_long_document_window_coverage_and_owner_weighting(self):
        vectors, cost = self.ns['embed_local'](['risk ' * 450, 'credit liquidity'])
        self.assertEqual([len(w.split()) for w in self.model.windows], [220, 220, 10, 2])
        self.assertEqual([int(t) for w in self.model.windows[:3] for t in w.split()], list(range(450)))
        raw = np.array([[n, 1, 2] for n in [220, 220, 10]], dtype=float)
        raw /= np.linalg.norm(raw, axis=1, keepdims=True)
        expected = np.average(raw, axis=0, weights=[220, 220, 10])
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(vectors[0], expected, rtol=1e-6)
        self.assertEqual(vectors.shape, (2, 3))
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-6)
        self.assertTrue(np.isfinite(vectors).all())
        self.assertEqual(cost, 0)

    def test_short_document_matches_direct_encode(self):
        actual, _ = self.ns['embed_local'](['credit risk'])
        expected = self.model.encode(['0 1'])
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_empty_batch_has_correct_shape_without_model_call(self):
        vectors, cost = self.ns['embed_local']([])
        self.assertEqual(vectors.shape, (0, 3))
        self.assertEqual(self.model.windows, [])
        self.assertEqual(cost, 0)

if __name__ == '__main__':
    unittest.main()
