"""Presentation backend, separate from the historical notebook benchmarks."""
import time
import uuid
from urllib.parse import urlparse

from src.evaluation_runtime import Runtime, citation_check
from src.research_pipeline import HybridResearch


def safe_source_url(value):
    return value if isinstance(value, str) and urlparse(value).scheme == 'https' else None


class DemoBackend:
    def __init__(self, base):
        self.runtime = Runtime(base, base / '.local' / 'demo' / uuid.uuid4().hex,
                               budget_usd=5, require_api_key=False)
        self.research = HybridResearch(self.runtime)

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
            api_cost_usd=sum(e.get('cost_usd', 0) for e in self.runtime.usage_for(case)))
        return result
