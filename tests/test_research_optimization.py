import contextlib
import copy
import threading
import unittest
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from src.demo_backend import DemoBackend
from src.evaluation_runtime import Runtime
from src.research_pipeline import HybridResearch


class OptimizationTests(unittest.TestCase):
    def backend(self):
        backend = DemoBackend.__new__(DemoBackend)
        backend.answer_cache = OrderedDict()
        backend.runtime = SimpleNamespace(corpus_hash='corpus-v1',
            scope=lambda *args: contextlib.nullcontext(), usage_for=lambda case: [{'cost_usd': .02}])
        backend.research = Mock()
        backend.research.answer.return_value = {'answer': 'TD quarterly average LCR was 138% [S1].',
            'sources': [{'bank_code': 'TD', 'fiscal_year': 2024}], 'evidence_checked': True}
        return backend

    def scope(self):
        return {'question': 'LCR TD 2024', 'scopes': [{'bank_code': 'TD', 'fiscal_year': 2024}],
                'answer_type': 'metric'}

    def test_cache_reuses_verified_result_without_charging_previous_cost(self):
        backend, scope = self.backend(), self.scope()
        first = backend.answer(scope)
        repeat = backend.answer(scope)
        self.assertFalse(first['cache_hit'])
        self.assertTrue(repeat['cache_hit'])
        self.assertEqual(repeat['api_cost_usd'], 0)
        backend.research.answer.assert_called_once()
        repeat['answer'] = 'changed'
        self.assertEqual(backend.answer(scope)['answer'], first['answer'])

    def test_cache_key_includes_corpus_and_validated_scope(self):
        backend, scope = self.backend(), self.scope()
        backend.answer(scope)
        backend.runtime.corpus_hash = 'corpus-v2'
        self.assertFalse(backend.answer(scope)['cache_hit'])
        other = copy.deepcopy(scope)
        other['scopes'][0]['bank_code'] = 'BMO'
        with self.assertRaises(ValueError):
            backend.answer(other)
        self.assertEqual(backend.research.answer.call_count, 3)

    def test_abstention_is_not_cached(self):
        backend, scope = self.backend(), self.scope()
        backend.research.answer.return_value['evidence_checked'] = False
        backend.answer(scope)
        backend.answer(scope)
        self.assertEqual(backend.research.answer.call_count, 2)

    def test_query_embedding_cache_keeps_vectors_private_and_reuses_encoding(self):
        runtime = Runtime.__new__(Runtime)
        runtime.provider = 'local'
        runtime.config = {'models': {'local': 'test-model'}}
        runtime.embed_lock = threading.Lock()
        encode = Mock(return_value=(np.array([[.5, .5]], dtype=np.float32), {}))
        runtime.embedding_ns, runtime.query_cache = {'embed_local': encode}, OrderedDict()
        runtime._record = Mock()
        runtime.embed('LCR')
        second = runtime.embed('LCR')
        second[0] = 99
        self.assertEqual(float(runtime.embed('LCR')[0]), .5)
        encode.assert_called_once()
        self.assertTrue(runtime._record.call_args.args[0]['cache_hit'])

    def test_review_retains_original_labels_and_only_cited_full_passages(self):
        hits = [{'chunk_id': f'chunk-{i}', 'text': f'Full passage {i} with table headers'} for i in range(3)]
        def context(hits):
            return '\n'.join(f'[S{i+1}] {h["text"]}' for i, h in enumerate(hits))
        pipeline = HybridResearch(SimpleNamespace(context=context))
        pipeline._json = Mock(return_value={'points': [], 'issues': []})
        pipeline.verify({'question': 'LCR?'}, [{'text': 'claim', 'evidence': [
            {'source': 'S3', 'quote': 'Full passage 2 with table headers'}]}], hits)
        payload = pipeline._json.call_args.args[3]
        self.assertEqual(payload['excerpts'], '[S3] Full passage 2 with table headers')
        schema = pipeline._json.call_args.args[1]
        enum = schema['points']['items']['properties']['evidence']['items']['properties']['source']['enum']
        self.assertEqual(enum, ['S3'])
