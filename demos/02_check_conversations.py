"""Opt-in live interpretation regression check. Makes paid API calls, no answer generation."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from src.conversation_interpreter import ConversationState, resolve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file', type=Path, default=ROOT / '.env')
    args = parser.parse_args()
    load_dotenv(args.env_file, encoding='utf-8-sig')
    records = []
    def turn(state, message, ready, banks=None, years=None, topic=None):
        result = resolve(message, state)
        records.append({'input': message, 'result': result})
        assert result['ready'] == ready, (message, result)
        if ready:
            assert set(result['banks']) == set(banks), (message, result)
            assert result['years'] == years, (message, result)
            if topic:
                assert topic.lower() in result['retrieval_query'].lower(), (message, result)
        print('PASS:', message, '| ready:', ready, '| banks:', state.banks, '| years:', state.years, flush=True)
        return result
    state = ConversationState()
    turn(state, "What did BMO disclose about credit risk in it's annual report?", False)
    turn(state, '2025', True, ['BMO'], [2025], 'credit risk')
    turn(state, "What was it's liquidity coverage ratio?", True, ['BMO'], [2025], 'liquidity')
    turn(state, 'Compare TD with it on liquidity coverage ratio.', True, ['TD','BMO'], [2025], 'liquidity')
    turn(state, 'What was its provision for credit losses?', False)
    turn(state, 'BMO', True, ['BMO'], [2025], 'credit')
    turn(state, 'What did bank of gotham disclose about credit risk in its 2025 annual report?', False)
    turn(state, 'What did Gotham disclose about credit risk in 2025?', False)
    turn(state, 'Actually, show RBC climate risk in 2024.', True, ['RBC'], [2024], 'climate')
    state = ConversationState()
    turn(state, 'What was the liquidity coverage ratio?', False)
    turn(state, '2025', False)
    turn(state, 'BMO', True, ['BMO'], [2025], 'liquidity')
    turn(state, 'What did RBC disclose in 2030?', False)
    output = ROOT / '.local' / 'conversation_live_check.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Saved', len(records), 'turns to', output, flush=True)


if __name__ == '__main__':
    main()
