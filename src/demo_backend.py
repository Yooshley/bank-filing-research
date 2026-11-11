"""Connect validated conversation scope to existing retrieval and agent tools."""
import time
import uuid
from urllib.parse import urlparse

from src.evaluation_runtime import Runtime, citation_check


def safe_source_url(value):
    return value if isinstance(value, str) and urlparse(value).scheme == 'https' else None


class DemoBackend:
    def __init__(self, base):
        self.runtime = Runtime(base, base / '.local' / 'demo' / uuid.uuid4().hex,
                               budget_usd=2, require_api_key=False)

    def answer(self, scope, mode):
        runtime = self.runtime
        case = uuid.uuid4().hex
        start = time.perf_counter()
        with runtime.scope(case, 'generation'):
            if mode == 'Agent':
                original = runtime.ns['FUNCTIONS']
                def guarded(name, function):
                    def call(**args):
                        bank = args.get('bank')
                        bank = 'BNS' if bank and bank.upper() in ('SCOTIABANK', 'BANK OF NOVA SCOTIA') else bank
                        if bank is None and len(scope['banks']) == 1:
                            bank = scope['banks'][0]
                        if bank not in scope['banks']:
                            raise ValueError('Name one of the validated banks in every tool call.')
                        args['bank'] = bank
                        report = scope.get('document_type')
                        if report and name in ('search_filings', 'compare_sections'):
                            if args.get('document_type') not in (None, report):
                                raise ValueError('Report type must match the validated request.')
                            args['document_type'] = report
                        if name == 'compare_sections':
                            if any(args.get(k) not in scope['years'] for k in ('year_a', 'year_b')):
                                raise ValueError('Comparison years must match validated scope.')
                        else:
                            year = args.get('fiscal_year')
                            if year is None and len(scope['years']) == 1:
                                year = scope['years'][0]
                            if year not in scope['years']:
                                raise ValueError('Specify a validated fiscal year.')
                            args['fiscal_year'] = year
                        return function(**args)
                    return call
                runtime.ns['FUNCTIONS'] = {n: guarded(n, f) for n, f in original.items()}
                try:
                    result = runtime.agent(scope)
                finally:
                    runtime.ns['FUNCTIONS'] = original
            else:
                result = runtime.rag(scope, structured=mode == 'Structured MMR')
        for hit in result['sources']:
            if hit.get('bank_code') not in scope['banks'] or hit.get('fiscal_year') not in scope['years']:
                raise ValueError('Retrieved evidence did not match the validated scope.')
            if scope.get('document_type') and hit.get('document_type') != scope['document_type']:
                raise ValueError('Retrieved report type did not match the validated request.')
        references = (result.get('structured') or {}).get('sources', [])
        citation_text = result['answer'] + '\n' + ' '.join('[' + s['label'] + ']' for s in references)
        checks = citation_check(citation_text, result['sources'])
        abstention = result['answer'].strip() == 'The provided filings do not contain enough information to answer this question.'
        result['cited_labels'] = [s['label'] for s in references]
        if result['sources'] and not abstention and (not checks['citation_identifiers_valid'] or not checks['citation_present']):
            result['answer'] = 'The answer did not pass citation checks. Please inspect the evidence or try a narrower question.'
        events = runtime.usage_for(case)
        result.update(seconds=round(time.perf_counter()-start, 2),
                      api_cost_usd=sum(e.get('cost_usd', 0) for e in events), checks=checks)
        return result
