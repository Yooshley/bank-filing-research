import argparse
from pathlib import Path
from src.evaluation import run_evaluation


def main():
    parser = argparse.ArgumentParser(description='Run and resume the Module 4 Step 6 evaluation')
    parser.add_argument('--live', action='store_true', help='Permit paid generation and judging calls')
    parser.add_argument('--smoke', action='store_true', help='One baseline question plus judge')
    parser.add_argument('--limit', type=int, default=30)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--budget-usd', type=float, default=5.0)
    args = parser.parse_args()
    result = run_evaluation(Path(__file__).resolve().parents[1], live=args.live, workers=args.workers,
                            budget_usd=args.budget_usd, limit=args.limit, smoke=args.smoke)
    print(result['summary'].to_string(index=False))
    print('Reports:', result['run_path'])
    if args.live and any(case['status']=='error' for case in result['cases']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
