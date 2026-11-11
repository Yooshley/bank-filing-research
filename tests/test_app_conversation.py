"""UI integration check with no model downloads, API calls or local index."""
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest
from src.conversation_interpreter import apply_interpretation


class AppConversationTests(unittest.TestCase):
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

        retrieve = Mock(return_value=[])
        backend = SimpleNamespace(runtime=SimpleNamespace(retrieve_scopes=retrieve))
        with patch('src.conversation_interpreter.resolve', side_effect=interpret), \
             patch('src.cloud_index.ensure_cloud_index'), \
             patch('src.demo_backend.DemoBackend', return_value=backend):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run()
            app.chat_input[0].set_value('BMO credit risk in its annual report').run()
            self.assertFalse(app.exception)
            retrieve.assert_not_called()
            app.chat_input[0].set_value('2025').run()
            self.assertFalse(app.exception)
            query, scopes, _ = retrieve.call_args.args
            self.assertIn('credit risk disclosures', query)
            self.assertEqual(scopes, [{'bank_code': 'BMO', 'fiscal_year': 2025,
                                       'document_type': 'annual_report'}])
            self.assertTrue(any(e.label == 'Understood research question' for e in app.expander))
