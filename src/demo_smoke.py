"""Exercise local retrieval and live conversational generation after rebuilding."""
import json
from pathlib import Path

from src.conversation_scope import ConversationState, resolve
from src.demo_backend import DemoBackend


def main():
    base = Path(__file__).resolve().parents[1]
    backend = DemoBackend(base)
    state = ConversationState()
    results = []
    for question, mode, banks in [
        ('What did BMO disclose about credit risk in 2025?', 'Structured MMR', ['BMO']),
        ('What was its provision for credit losses?', 'Structured MMR', ['BMO']),
        ('Compare TD with it on credit risk disclosures.', 'Agent', ['TD', 'BMO']),
    ]:
        scope = resolve(question, state)
        assert scope['ready'] and scope['banks'] == banks, scope
        print('Testing:', question, flush=True)
        result = backend.answer(scope, mode)
        assert result['answer'] and result['sources'], 'Missing answer or evidence'
        assert result['checks']['citation_identifiers_valid'], 'Invalid citations'
        assert not result['answer'].startswith('The answer did not pass citation checks'), 'Answer was rejected'
        state.accept(scope)
        results.append({'question': question, 'mode': mode, 'scope': scope, 'result': result})
        print('Completed:', len(result['sources']), 'sources;', result['seconds'], 'seconds', flush=True)
    assert not resolve('What did Bank of Atlantis report in 2025?', state)['ready']
    output = base / '.local' / 'demo_smoke.json'
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print('Saved live test evidence:', output, flush=True)


if __name__ == '__main__':
    main()
