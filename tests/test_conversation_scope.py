import unittest
from src.conversation_scope import ConversationState, resolve


class ConversationTests(unittest.TestCase):
    def setUp(self):
        self.state = ConversationState()

    def accept(self, text):
        result = resolve(text, self.state)
        self.assertTrue(result['ready'], result)
        self.state.accept(result)
        return result

    def test_followup_switch_and_comparison(self):
        self.accept('BMO credit risk in 2025')
        self.assertEqual(self.accept('What were its credit losses?')['banks'], ['BMO'])
        self.assertEqual(self.accept('Compare TD with it')['banks'], ['TD', 'BMO'])
        self.assertFalse(resolve('What were its losses?', self.state)['ready'])
        self.assertEqual(self.accept('Now describe TD liquidity risk')['banks'], ['TD'])
        self.assertEqual(self.accept('What was its LCR?')['banks'], ['TD'])
        self.assertEqual(self.accept('Compare TD with it')['banks'], ['TD', 'BMO'])

    def test_unknown_bank_cannot_inherit_known_entity(self):
        self.accept('BMO credit risk in 2025')
        self.assertFalse(resolve('What did Bank of Atlantis report?', self.state)['ready'])
        self.assertEqual(self.state.banks, ['BMO'])

    def test_no_year_and_out_of_corpus_year(self):
        self.assertFalse(resolve('BMO credit risk', self.state)['ready'])
        self.assertFalse(resolve('BMO credit risk 2030', self.state)['ready'])

    def test_year_override_and_two_period_comparison(self):
        self.accept('BMO liquidity in 2025')
        result = self.accept('Compare BMO liquidity in 2024 and 2025')
        self.assertEqual(result['years'], [2024, 2025])
        self.assertEqual(len(result['scopes']), 2)

    def test_alias_and_explicit_possessive(self):
        self.assertEqual(self.accept('What did Scotiabank say about its risks in 2025?')['banks'], ['BNS'])


if __name__ == '__main__':
    unittest.main()
