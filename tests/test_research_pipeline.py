import unittest
import threading
from types import SimpleNamespace
from unittest.mock import Mock

from src.research_pipeline import HybridResearch, lexical_rank, literal_support
from src.evaluation_runtime import token_cost, PRICES
from src.evaluation_runtime import Runtime, BudgetExceeded


def hit(key, text, bank='TD', section='Liquidity'):
    return {'chunk_id': key, 'text': text, 'bank_code': bank, 'fiscal_year': 2024,
            'document_type': 'annual_report', 'section': section}


class ResearchTests(unittest.TestCase):
    def test_budget_reservation_uses_selected_reasoning_model_prices(self):
        runtime = Runtime.__new__(Runtime)
        runtime.client = Mock()
        runtime.model = 'gpt-4o-mini'
        runtime.lock, runtime.ledger, runtime.reserved = threading.RLock(), [], 0
        runtime.budget = .001
        with self.assertRaises(BudgetExceeded):
            runtime.chat(model='gpt-6-astra', max_completion_tokens=100,
                         messages=[{'role': 'user', 'content': 'Test'}])
        runtime.client.chat.completions.create.assert_not_called()

    def test_reasoning_model_cost_includes_cache_writes(self):
        measured = token_cost(1000, 50, 200, PRICES['gpt-6-astra'], 100)
        self.assertAlmostEqual(measured, .01095)

    def test_keyword_search_finds_metric_and_penalizes_glossary(self):
        factual = hit('fact', 'The average liquidity coverage ratio was 138% for the quarter.')
        glossary = hit('definition', 'The liquidity coverage ratio measures liquidity.', section='GLOSSARY')
        irrelevant = hit('other', 'We adopted new financial reporting standards.')
        result = lexical_rank('liquidity coverage ratio', [irrelevant, glossary, factual])
        self.assertEqual(result[0]['chunk_id'], 'fact')
        self.assertNotIn(irrelevant, result)

    def test_exact_quote_and_numbers_must_be_present(self):
        hits = [hit('fact', 'The average LCR was 138% for the quarter ended October 31, 2024.')]
        def point(text, quote, label='S1'):
            return {'text': text, 'evidence': [{'source': label, 'quote': quote}]}
        quote = hits[0]['text']
        accepted = point('The quarterly average LCR was 138%.', quote)
        cases = [accepted, point('The LCR was 150%.', quote),
                 point('The LCR was 138%.', 'The reported LCR was 138%.'),
                 point('The LCR was 138%.', quote, 'S9')]
        self.assertEqual(literal_support(cases, hits), [accepted])

    def test_known_chunk_ids_and_ordered_quotation_spans_are_resolved(self):
        text = 'The quarterly average LCR was 138%. Additional explanation. It exceeded the required minimum.'
        points = [{'text': 'The quarterly average LCR was 138%.', 'evidence': [{
            'source': 'td', 'quote': 'The quarterly average LCR was 138%...It exceeded the required minimum.'}]}]
        result = literal_support(points, [hit('td', text)])
        self.assertEqual(result[0]['evidence'][0]['source'], 'S1')
        points[0]['evidence'][0]['quote'] = 'It exceeded the required minimum...The quarterly average LCR was 138%.'
        self.assertEqual(literal_support(points, [hit('td', text)]), [])

    def test_metadata_can_establish_fiscal_year_but_not_a_value(self):
        text = 'Provision was $100 million compared with $80 million in the prior year.'
        points = [{'text': 'Provision in 2024 was $100 million, up from $80 million in 2023.',
                   'evidence': [{'source': 'S1', 'quote': text}]}]
        self.assertEqual(len(literal_support(points, [hit('td', text)])), 1)
        points[0]['text'] = 'Provision was $200 million.'
        self.assertEqual(literal_support(points, [hit('td', text)]), [])

    def test_hybrid_paths_cannot_cross_scope(self):
        td = hit('td', 'Average liquidity coverage ratio 138%')
        bmo = hit('bmo', 'Average liquidity coverage ratio 132%', bank='BMO')
        runtime = SimpleNamespace(corpus={'td': td, 'bmo': bmo}, search=Mock(return_value=[bmo, td]))
        scope = {'retrieval_query': 'liquidity coverage ratio',
                 'scopes': [{'bank_code': 'TD', 'fiscal_year': 2024, 'document_type': 'annual_report'}]}
        hits, _ = HybridResearch(runtime).retrieve(scope)
        self.assertEqual([h['chunk_id'] for h in hits], ['td'])

    def test_review_can_reject_a_claim_despite_exact_quote(self):
        # The number exists, but the model changed the quarterly average into a year-end value.
        text = 'The average LCR was 138% for the quarter ended October 31, 2024.'
        points = [{'text': 'The year-end LCR was 138%.',
                   'evidence': [{'source': 'S1', 'quote': text}]}]
        pipeline = HybridResearch(SimpleNamespace(context=lambda hits: text))
        pipeline._json = Mock(return_value={'points': [], 'issues': ['Incorrect period basis']})
        accepted, issues = pipeline.verify({'question': 'What was the LCR?'}, points, [hit('td', text)])
        self.assertEqual(accepted, [])
        self.assertTrue(issues)

    def test_sources_are_renumbered_after_evidence_selection(self):
        text = 'The average LCR was 138% for the quarter ended October 31, 2024.'
        hits = [hit('irrelevant', 'Other evidence.'), hit('td', text)]
        pipeline = HybridResearch(None)
        pipeline.retrieve = Mock(return_value=(hits, []))
        point = {'text': 'S2: TD’s quarterly average LCR was 138%. [S2]',
                 'evidence': [{'source': 'S2', 'quote': text}]}
        pipeline.draft = Mock(return_value={'points': [point], 'limitation': '', 'next_query': ''})
        pipeline.verify = Mock(return_value=([point], []))
        result = pipeline.answer({'question': 'LCR?'})
        self.assertIn('[S1]', result['answer'])
        self.assertNotIn('[S2]', result['answer'])
        self.assertNotIn('S2:', result['answer'])
        self.assertEqual(result['sources'], [hits[1]])

    def test_unverified_facts_produce_abstention(self):
        hits = [hit('td', 'A definition without a reported value.')]
        pipeline = HybridResearch(None)
        pipeline.retrieve = Mock(return_value=(hits, []))
        pipeline.draft = Mock(return_value={'points': [], 'limitation': '', 'next_query': ''})
        result = pipeline.answer({'question': 'LCR?'})
        self.assertFalse(result['evidence_checked'])
        self.assertIn('couldn’t verify', result['answer'])
