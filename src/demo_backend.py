"""Presentation backend, separate from the historical notebook benchmarks."""
import time
import uuid
import copy
from collections import OrderedDict
from urllib.parse import urlparse

from src.evaluation_runtime import Runtime, citation_check, fingerprint
from src.research_pipeline import HybridResearch, MODEL


def load_search_resources(base):
    """Share only the index/model; never API clients, budgets or user dialogue."""
    runtime = Runtime(base, base / '.local' / 'search_bootstrap', require_api_key=False)
    names = ('config', 'provider', 'store', 'collection', 'corpus', 'corpus_hash',
             'filings', 'embedding_ns', 'embed_lock')
    return {name: getattr(runtime, name) for name in names}


def safe_source_url(value):
    return value if isinstance(value, str) and urlparse(value).scheme == 'https' else None


class DemoBackend:
    def __init__(self, base, search_resources=None):
        self.runtime = Runtime(base, base / '.local' / 'demo' / uuid.uuid4().hex,
                               budget_usd=5, require_api_key=False, search_resources=search_resources)
        self.research = HybridResearch(self.runtime)
        self.runtime.query_cache = OrderedDict()
        self.answer_cache = OrderedDict()

    def retrieve(self, scope):
        started = time.perf_counter()
        case = uuid.uuid4().hex
        with self.runtime.scope(case, 'generation'):
            hits, trace = self.research.retrieve(scope)
        return {'answer': 'Retrieved filing evidence is shown below.', 'sources': hits,
                'trace': trace, 'seconds': round(time.perf_counter()-started, 2),
                'api_cost_usd': sum(e.get('cost_usd', 0) for e in self.runtime.usage_for(case))}

    def answer(self, scope, mode='Research assistant'):
        case, started = uuid.uuid4().hex, time.perf_counter()
        key = fingerprint({'corpus': self.runtime.corpus_hash, 'model': MODEL, 'version': 2,
                           'mode': mode, 'question': scope['question'],
                           'scopes': scope['scopes'], 'answer_type': scope.get('answer_type')})
        if key in self.answer_cache:
            self.answer_cache.move_to_end(key)
            result = copy.deepcopy(self.answer_cache[key])
            result.update(cache_hit=True, api_cost_usd=0.0, seconds=round(time.perf_counter()-started, 3))
            return result
        with self.runtime.scope(case, 'generation'):
            if mode == 'Basic top-k (baseline)':
                result = self.runtime.rag(scope, structured=False)
                checks = citation_check(result['answer'], result['sources'])
                if not checks['citation_identifiers_valid']:
                    result['answer'] = 'The answer did not pass citation checks. Please inspect the evidence.'
            else:
                result = self.research.answer(scope)
        for hit in result['sources']:
            if not any(all(hit.get(k) == v for k, v in s.items()) for s in scope['scopes']):
                raise ValueError('Retrieved evidence did not match the validated request.')
        result.update(seconds=round(time.perf_counter()-started, 2),
            api_cost_usd=sum(e.get('cost_usd', 0) for e in self.runtime.usage_for(case)), cache_hit=False)
        if result.get('evidence_checked'):
            self.answer_cache[key] = copy.deepcopy(result)
            if len(self.answer_cache) > 64:
                self.answer_cache.popitem(last=False)
        return result
