"""UI integration check with no model downloads, API calls or local index."""
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest
from src.conversation_interpreter import apply_interpretation


class AppConversationTests(unittest.TestCase):
    def test_generated_answers_are_the_default_workflow(self):
        scope = {'ready': True, 'question': 'TD liquidity coverage ratio', 'original': 'LCR?',
                 'retrieval_query': 'liquidity coverage ratio', 'banks': ['TD'], 'years': [2024],
                 'scopes': [{'bank_code': 'TD', 'fiscal_year': 2024}], 'interpretation_cost_usd': .01}
        answer = Mock(return_value={'answer': 'TD’s quarterly average LCR was 138%.',
                                    'sources': [], 'seconds': 1, 'api_cost_usd': .2})
        with patch('src.conversation_interpreter.resolve', return_value=scope), \
             patch('src.cloud_index.ensure_cloud_index'), \
             patch('src.demo_backend.DemoBackend', return_value=SimpleNamespace(answer=answer)):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run()
            self.assertFalse(app.checkbox[0].value)
            app.chat_input[0].set_value('LCR?').run()
            self.assertFalse(app.exception)
            answer.assert_called_once_with(scope, 'Research assistant')
            self.assertTrue(any('138%' in m.value for m in app.markdown))

    def test_year_reply_retrieves_original_request(self):
        def interpret(message, state):
            return apply_interpretation(message, state, {
                'action': 'clarify' if message != '2025' else 'search',
                'turn_type': 'new_request' if message != '2025' else 'clarification',
                'topic': 'credit risk disclosures' if message != '2025' else '2025',
                'standalone_question': '', 'banks': ['BMO'],
                'years': [] if message != '2025' else [2025],
                'document_type': 'annual_report',
                'missing_fields': ['year'] if message != '2025' else [],
                'clarification_question': '', 'bank_mentions': []})

        retrieve = Mock(return_value={'answer': 'Evidence', 'sources': [], 'seconds': 0, 'api_cost_usd': 0})
        backend = SimpleNamespace(retrieve=retrieve)
        with patch('src.conversation_interpreter.resolve', side_effect=interpret), \
             patch('src.cloud_index.ensure_cloud_index'), \
             patch('src.demo_backend.DemoBackend', return_value=backend):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run()
            app.checkbox[0].set_value(True).run()
            app.chat_input[0].set_value('BMO credit risk in its annual report').run()
            self.assertFalse(app.exception)
            retrieve.assert_not_called()
            app.chat_input[0].set_value('2025').run()
            self.assertFalse(app.exception)
            scope = retrieve.call_args.args[0]
            self.assertIn('credit risk disclosures', scope['retrieval_query'])
            self.assertEqual(scope['scopes'], [{'bank_code': 'BMO', 'fiscal_year': 2025,
                                       'document_type': 'annual_report'}])
            self.assertTrue(any(e.label == 'Understood research question' for e in app.expander))
