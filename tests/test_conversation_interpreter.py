import copy
import json
import unittest
from types import SimpleNamespace

from src.conversation_interpreter import ConversationState, apply_interpretation, resolve


def request(**changes):
    value = {'action': 'search', 'turn_type': 'new_request', 'topic': 'credit risk disclosures',
             'standalone_question': 'BMO credit risk disclosures in 2025', 'banks': ['BMO'],
             'years': [2025], 'document_type': 'annual_report', 'missing_fields': [],
             'clarification_question': '', 'bank_mentions': []}
    value.update(changes)
    return value


class InterpreterTests(unittest.TestCase):
    def test_context_does_not_allow_invented_scope_changes(self):
        state = ConversationState(banks=['BMO'], years=[2025])
        for value in [request(banks=['TD']), request(years=[2024])]:
            with self.assertRaises(ValueError):
                apply_interpretation('What was its liquidity coverage ratio?', state, value)

    def test_bare_bank_reply_fills_pending_even_if_model_omits_it(self):
        state = ConversationState()
        apply_interpretation('liquidity coverage ratio in 2025', state,
                             request(action='clarify', banks=[], missing_fields=['bank'],
                                     topic='liquidity coverage ratio'))
        result = apply_interpretation('BMO', state, request(action='clarify', banks=[],
                                      topic='', missing_fields=['bank']))
        self.assertTrue(result['ready'])
        self.assertIn('liquidity coverage ratio', result['question'])

    def test_model_cannot_invent_initial_bank_or_year(self):
        state = ConversationState()
        result = apply_interpretation('BMO credit risk', state, request())
        self.assertFalse(result['ready'])
        self.assertEqual(state.years, [])
        state = ConversationState()
        result = apply_interpretation('credit risk in 2025', state, request())
        self.assertFalse(result['ready'])
        self.assertEqual(state.banks, [])

    def test_year_clarification_keeps_bank_topic_and_report(self):
        state = ConversationState()
        first = request(action='clarify', years=[], missing_fields=['year'],
                        bank_mentions=[{'text': 'BMO', 'bank': 'BMO'}])
        self.assertFalse(apply_interpretation('BMO credit risk in its annual report', state, first)['ready'])
        self.assertEqual(state.pending_request['topic'], 'credit risk disclosures')
        # Even if a model drops other slots while filling the year, restore them.
        second = request(turn_type='clarification', banks=[], document_type='', topic='2025')
        result = apply_interpretation('2025', state, second)
        self.assertTrue(result['ready'])
        self.assertEqual(result['banks'], ['BMO'])
        self.assertEqual(result['years'], [2025])
        self.assertIn('credit risk disclosures', result['retrieval_query'])
        self.assertEqual(result['document_type'], 'annual_report')
        self.assertIsNone(state.pending_request)
        self.assertIn('Which fiscal year', state.history[1]['content'])

    def test_sequential_year_and_bank_clarifications(self):
        state = ConversationState()
        apply_interpretation('liquidity coverage ratio', state, request(
            action='clarify', banks=[], years=[], topic='liquidity coverage ratio',
            missing_fields=['bank', 'year']))
        middle = apply_interpretation('2025', state, request(action='clarify',
            turn_type='clarification', banks=[], topic='', missing_fields=['bank']))
        self.assertFalse(middle['ready'])
        final = apply_interpretation('BMO', state, request(turn_type='clarification',
            topic='BMO', bank_mentions=[{'text': 'BMO', 'bank': 'BMO'}]))
        self.assertTrue(final['ready'])
        self.assertIn('liquidity coverage ratio', final['question'])

    def test_comparison_and_ambiguous_followup(self):
        state = ConversationState()
        apply_interpretation('BMO credit risk in 2025', state, request())
        comparison = apply_interpretation('Compare TD with it', state, request(
            turn_type='follow_up', banks=['TD', 'BMO'], bank_mentions=[{'text': 'TD', 'bank': 'TD'}]))
        self.assertEqual({s['bank_code'] for s in comparison['scopes']}, {'BMO', 'TD'})
        ambiguous = apply_interpretation('What was its liquidity coverage ratio?', state,
            request(action='search', turn_type='follow_up', banks=['BMO'],
                    topic='liquidity coverage ratio', missing_fields=[]))
        self.assertFalse(ambiguous['ready'])
        final = apply_interpretation('BMO', state, request(turn_type='clarification', topic='BMO'))
        self.assertTrue(final['ready'])
        self.assertIn('liquidity coverage ratio', final['question'])

    def test_new_request_does_not_inherit_pending_topic(self):
        state = ConversationState()
        apply_interpretation('BMO credit risk', state, request(action='clarify', years=[], missing_fields=['year']))
        result = apply_interpretation('RBC climate risk in 2024', state, request(
            topic='climate risk', banks=['RBC'], years=[2024]))
        self.assertTrue(result['ready'])
        self.assertNotIn('credit risk', result['question'])

    def test_unknown_named_bank_cannot_be_substituted_by_model(self):
        for message in ['bank of gotham', 'BANK OF GOTHAM', 'Gotham Bank', 'Bank of Atlantis']:
            state = ConversationState(banks=['BMO'], years=[2025])
            before = copy.deepcopy(state)
            self.assertFalse(apply_interpretation(message, state, request())['ready'])
            self.assertEqual(state, before)

    def test_unrecognized_mention_and_alias_mapping(self):
        state = ConversationState()
        result = apply_interpretation('Gotham credit risk', state, request(
            bank_mentions=[{'text': 'Gotham', 'bank': 'BMO'}]))
        self.assertFalse(result['ready'])
        with self.assertRaises(ValueError):
            apply_interpretation('Scotiabank credit risk in 2025', state,
                request(bank_mentions=[{'text': 'Scotiabank', 'bank': 'BMO'}]))
        self.assertFalse(state.history)

    def test_explicit_year_and_bank_cannot_be_dropped(self):
        for message in ['TD credit risk in 2025', 'BMO credit risk in 2024']:
            with self.assertRaises(ValueError):
                apply_interpretation(message, ConversationState(), request())

    def test_out_of_corpus_year_does_not_mutate_state(self):
        state = ConversationState()
        self.assertFalse(apply_interpretation('BMO in 2030', state, request(years=[2030]))['ready'])
        self.assertFalse(state.history)

    def test_model_call_uses_history_schema_and_tracks_cost(self):
        calls = []
        def create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',
                message=SimpleNamespace(content=json.dumps(request())))],
                usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50,
                                      prompt_tokens_details=None))
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        state = ConversationState()
        result = resolve('BMO credit risk in 2025', state, client)
        self.assertTrue(result['ready'])
        self.assertGreater(result['interpretation_cost_usd'], 0)
        self.assertTrue(calls[0]['response_format']['json_schema']['strict'])
        self.assertIn('pending_request', json.loads(calls[0]['messages'][1]['content']))

    def test_truncation_fails_closed(self):
        def create(**kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(finish_reason='length')], usage=None)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        state = ConversationState()
        with self.assertRaises(ValueError):
            resolve('BMO credit risk 2025', state, client)
        self.assertFalse(state.history)
