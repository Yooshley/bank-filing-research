"""Opt-in paid end-to-end quality check against an existing local filing index."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
from src.conversation_interpreter import ConversationState, resolve
from src.demo_backend import DemoBackend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT)
    parser.add_argument('--env-file', type=Path, default=ROOT / '.env')
    args = parser.parse_args()
    load_dotenv(args.env_file, encoding='utf-8-sig')
    output = ROOT / '.local' / 'research_live_check.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    backend, state, records = DemoBackend(args.data_root.resolve()), ConversationState(), []
    cases = [
        ('what did bmo disclose about credit risk in its 2024 annual report?', ['BMO'], [2024], None),
        ('what about TD?', ['TD'], [2024], None),
        ('what was its liquidity coverage ratio?', ['TD'], [2024], '138'),
        ('Compare it with BMO on that ratio.', ['TD', 'BMO'], [2024], '132'),
    ]
    for question, banks, years, expected in cases:
        started = time.perf_counter()
        scope = resolve(question, state)
        assert scope['ready'], (question, scope)
        assert set(scope['banks']) == set(banks), (question, scope)
        assert scope['years'] == years, (question, scope)
        answer = backend.answer(scope)
        record = {'question': question, 'scope': scope, 'result': answer,
                  'elapsed_seconds': round(time.perf_counter()-started, 2)}
        records.append(record)
        output.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding='utf-8')
        print(question, '\n', answer['answer'], '\n', 'seconds:', record['elapsed_seconds'],
              'cost:', answer['api_cost_usd'], flush=True)
        assert answer.get('evidence_checked'), (question, answer)
        assert len(answer['answer'].split()) <= 200, (question, answer)
        if expected:
            assert expected in answer['answer'], (question, answer)
            assert 'quarter' in answer['answer'].lower(), (question, answer)
        if len(banks) > 1:
            assert {h['bank_code'] for h in answer['sources']} == set(banks)
    print('PASS: 4 full research answers; checkpoint:', output, flush=True)


if __name__ == '__main__':
    main()
