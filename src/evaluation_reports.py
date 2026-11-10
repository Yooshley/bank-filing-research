"""Write the three curriculum reports from recorded measurements only."""
import csv
import json
from pathlib import Path

import pandas as pd


def write_reports(base, run_path, manifest, cases, summary, runtime):
    reports = base / 'reports'
    reuse_path = run_path / 'reuse_provenance.json'
    reuse = json.loads(reuse_path.read_text(encoding='utf-8')) if reuse_path.exists() else None
    reuse_note = (f"This run reuses {len(reuse['reused_cases'])} unaffected completed baseline/MMR records from "
                  f"`{reuse['parent_run']}` after confirming identical corpus/test fingerprints and exact source metadata. "
                  'All changed agent cases and previously rejected structured cases were generated and judged again. '
                  'Each reused case records its origin_case_id and reused_from_run; the original API ledger remains the cost evidence. '
                  'See reuse_provenance.json for the explicit lineage.' if reuse else
                  'Completed checkpoints are reused only within this evaluation identity.')
    complete = bool(len(summary) == 3 and summary['complete'].all())
    scored = [case for case in cases if case.get('status') == 'complete']
    candidates = summary[summary['complete']]
    best = None
    if not candidates.empty:
        best = candidates.sort_values(['faithfulness', 'relevance', 'completeness'], ascending=False).iloc[0]
    totals = {'operating_usd': sum(case.get('generation_cost_usd', 0)+case.get('embedding_cost_usd',0) for case in cases),
              'judge_usd': sum(case.get('judge_cost_usd',0) for case in cases),
              'completed_answers': len(scored), 'expected_answers': 90,
              'status': 'complete' if complete else 'incomplete', 'recommendation': 'iterate'}
    (run_path/'run_summary.json').write_text(json.dumps(totals, indent=2), encoding='utf-8')
    # Preserve human labels. The automated judge is never presented as manual review.
    human_path = run_path / 'human_review.csv'
    previous = {}
    if human_path.exists():
        previous = {row['case_id']: row for row in csv.DictReader(human_path.open(encoding='utf-8-sig'))}
    human = []
    for case in scored:
        row = {'case_id': case['case_id'], 'config': case['config'], 'question_id': case['question']['id'],
               'question': case['question']['question'], 'answer': case['generation']['answer'],
               'human_faithfulness': '', 'human_relevance': '', 'human_completeness': '', 'notes': ''}
        for field in ['human_faithfulness','human_relevance','human_completeness','notes']:
            row[field] = previous.get(row['case_id'], {}).get(field, '')
        human.append(row)
    if human:
        pd.DataFrame(human).to_csv(human_path,index=False)
    hallucinations = [case for case in scored if case['judgment']['faithfulness'] <= 2]
    standard_refusal = 'The provided filings do not contain enough information to answer this question.'
    inconsistent_judgments = [case for case in scored
        if case['generation']['answer'].strip() == standard_refusal and case['judgment']['faithfulness'] < 5]
    consistency_rows = [{'case_id': case['case_id'], 'config': case['config'],
                         'question_id': case['question']['id'], 'faithfulness': case['judgment']['faithfulness'],
                         'reason': case['judgment']['faithfulness_reason'],
                         'issue': 'Fixed abstention penalized for faithfulness, contrary to the stated rubric. Review relevance/completeness separately.'}
                        for case in inconsistent_judgments]
    pd.DataFrame(consistency_rows, columns=['case_id','config','question_id','faithfulness','reason','issue']).to_csv(
        run_path/'judge_consistency_review.csv',index=False)
    analysis = ['# Hallucination case analysis', '',
                'Faithfulness ≤2 flags a material grounding risk under the LLM rubric. These cases require human confirmation.', '']
    for case in hallucinations:
        judgment = case['judgment']
        evidence = case['generation']['sources']
        if case['question']['id'] == 'eval_26' and 'Scotiabank' in case['generation']['answer']:
            cause = ('Observed routing failure: the question names unsupported Bank of Atlantis, but the agent answered '
                     'with Scotiabank figures. An unrestricted tool search substituted another bank. Add an explicit '
                     'unknown-bank gate and verify the final answer entity against the requested bank before returning it.')
        elif case['question']['id'] == 'eval_28':
            cause = ('Ambiguity handling needs review: the request specifies neither bank nor period. Retrieval and '
                     'synthesis can select an arbitrary institution instead of asking for clarification. Add a '
                     'clarification route before retrieval and assess the saved answer against this expected behavior.')
        else:
            cause = ('Likely cause to investigate: unsupported synthesis, incomplete multi-period/bank retrieval, '
                     'or mismatched numerical periods. ')
        analysis.extend([f"## {case['config']} {case['question']['id']}", '',
            f"Question: {case['question']['question']}", '',
            f"Faithfulness: {judgment['faithfulness']}/5. {judgment['faithfulness_reason']}", '',
            'Unsupported claims: ' + '; '.join(judgment['unsupported_claims']), '',
            'Evidence inspected: ' + ', '.join(source['chunk_id'] for source in evidence), '',
            ('Judge inconsistency: this is the fixed insufficient-information refusal, with no affirmative financial claim. '
             'The rubric assigns a faithful refusal 5 even when completeness is low. Treat this flag as a likely grader false positive; '
             'retain the raw score and request independent adjudication.'
             if case in inconsistent_judgments else
             cause)
            +
            'The judge explanation identifies the specific claim; confirm it against the saved context before assigning a definitive cause.', '',
            'Remediation: require supporting citations for each disputed claim, retrieve missing banks/periods, '
            'and use deterministic arithmetic with cited inputs; abstain when either input is absent.', ''])
    if not hallucinations:
        analysis += ['No scored case currently has faithfulness ≤2. This does not establish absence of hallucinations outside this sample.', '']
    (run_path/'hallucination_analysis.md').write_text('\n'.join(analysis),encoding='utf-8')
    # Charts are standard notebook/repository artifacts, not synthetic scores.
    if scored:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        metric_columns = ['faithfulness','relevance','completeness']
        available = summary.dropna(subset=metric_columns) if all(c in summary for c in metric_columns) else pd.DataFrame()
        if not available.empty:
            ax = available.set_index('config')[metric_columns].plot.bar(figsize=(10,4),ylim=(0,5),rot=15)
            ax.set(ylabel='LLM-judge score (1–5)',title='Shared 30-question quality comparison')
            ax.figure.tight_layout();ax.figure.savefig(run_path/'quality_comparison.png',dpi=160);plt.close(ax.figure)
            fig,axes=plt.subplots(1,2,figsize=(10,4))
            available.plot.bar(x='config',y='mean_cost_per_query_usd',ax=axes[0],legend=False,rot=15)
            axes[0].set(ylabel='Estimated USD',title='Operating API cost per query')
            available.plot.bar(x='config',y='mean_latency_seconds',ax=axes[1],legend=False,rot=15)
            axes[1].set(ylabel='Seconds',title='Observed mean latency')
            fig.tight_layout();fig.savefig(run_path/'cost_latency.png',dpi=160);plt.close(fig)
    quality_text = ('No full configuration has completed all 30 judgments yet.' if best is None else
        f"The strongest complete configuration by faithfulness, relevance and completeness is {best['config']}: "
        f"{best['faithfulness']:.2f}/5 faithfulness, {best['relevance']:.2f}/5 relevance, "
        f"and {best['completeness']:.2f}/5 completeness. Its estimated operating API cost is "
        f"US${best['mean_cost_per_query_usd']:.6f} per query, or US${best['annual_cost_1000_per_day_usd']:.2f} "
        'per year at 1,000 queries every day.')
    executive = f'''# Executive summary

Recommendation: **iterate further under analyst supervision**. The system searches public Canadian bank filings and drafts cited answers. It is a research aid; final compliance conclusions remain with an analyst.

The evaluation is {totals['status']}: {len(scored)} of 90 planned answers have been judged, using the same 30 questions across basic retrieval, diverse retrieval with structured output, and a tool-using agent. {quality_text}

The automated judge flagged {len(hallucinations)} answers with faithfulness at or below 2/5. These are unconfirmed flags; {len(inconsistent_judgments)} fixed-refusal scores also conflict with the stated grading rubric. Review these cases before any deployment. Human accuracy and citation reviews from steps 4 and 5 are still unfilled, and automated scores do not replace them. The judge uses the same model family as the answer generator, so correlated errors are possible.

The corpus contains {len(runtime.filings)} distinct filings and {len(runtime.corpus):,} indexed chunks. Some quarterly coverage is missing; a claim about a new risk cannot be established when an earlier filing is absent. Search excerpts cannot prove a topic is absent from an entire report.

The cost estimate covers measured API tokens. It excludes hosting, local computation, analyst review and setup. No timed analyst study has been performed, so the proposed 50% reduction in review time has not been demonstrated. Complete human checks, repair low-faithfulness cases, fill data gaps and measure analyst task time before a supervised pilot.

Detailed results: [evaluation report](evaluation_report.md), [system documentation](system_documentation.md), and [master comparison](evaluation/{manifest['run_id']}/master_comparison.csv).
'''
    (reports/'executive_summary.md').write_text(executive,encoding='utf-8')
    comparison = []
    for row in summary.to_dict('records'):
        if not row['scored_questions']:
            comparison.append(f"- {row['config']}: no completed judge scores; {row['error_questions']} errors.")
            continue
        comparison.append(f"- {row['config']}: {row['scored_questions']}/30 scored; faithfulness {row['faithfulness']:.2f}/5, relevance {row['relevance']:.2f}/5, completeness {row['completeness']:.2f}/5; "
            f"{row['hallucination_risk_count']} grounding-risk cases; mean latency {row['mean_latency_seconds']:.2f}s (p95 {row['p95_latency_seconds']:.2f}s); "
            f"operating cost US${row['mean_cost_per_query_usd']:.6f}/query; annual projection US${row['annual_cost_1000_per_day_usd']:.2f}; "
            f"quality targets {'met' if row['meets_quality_target'] else 'not met or incomplete'}.")
    evaluation = f'''# Evaluation report

Evaluation identifier: `{manifest['run_id']}`. Status: **{totals['status']}**, {len(scored)}/90 answers judged. Recommendation: **iterate**.

## Method and test coverage

Thirty fixed questions cover all five banks, credit/market/operational/liquidity/climate/model risk, and factual (15), comparative (5), trend (5) and edge-case (5) requests. See [test set](evaluation/{manifest['run_id']}/test_set.json). Every configuration receives identical questions and bank/year requirements. The agent chooses its own tools; baseline/MMR retrieval reserves evidence for requested scopes rather than allowing one bank to consume all result slots. Each non-agent question has an eight-chunk total budget; MMR selects from up to 30 candidates per scope with lambda 0.7. Those settings are logged as parameters.

The three configurations are basic top-k with the step 4 basic prompt, MMR with schema-constrained answer/sources/confidence/caveats, and the step 5 tool-calling agent. The shared notebook definitions are loaded without executing install/example cells. The index and embedding preprocessing from notebook 2 are retained. Corpus fingerprint: `{manifest['corpus_hash']}`. Test fingerprint: `{manifest['test_set_hash']}`.

{reuse_note}

## Master comparison

{chr(10).join(comparison)}

The complete machine-readable comparison is [master_comparison.csv](evaluation/{manifest['run_id']}/master_comparison.csv); individual scores are [per_question_metrics.csv](evaluation/{manifest['run_id']}/per_question_metrics.csv). Quality targets from the curriculum are faithfulness ≥3.5/5 and relevance ≥4.0/5. A configuration is marked complete only after all 30 answers are generated and judged. Errors remain visible and are not converted to zero scores or discarded without a count.

## Judge and grounding limitations

The LLM judge scores faithfulness, relevance and completeness on 1–5 integer rubrics with reasons and unsupported-claim lists. Faithfulness is checked against the exact evidence available to each answer; completeness additionally uses the same fixed reference packet for all configurations. Appropriate abstentions can be faithful but incomplete when reference evidence supports an answer. Unknown-bank/future/ambiguous questions permit a complete refusal or clarification. The judge is blinded to configuration names. Both generator and judge are `{manifest['answer_model']}`; self-evaluation bias remains. Reference packets are retrieved evidence, not expert-written gold answers or exhaustive whole-document review. Human ratings are preserved separately and remain required for assurance.

Citation checks validate source identifiers and structured metadata; they do not prove that every claim is supported. Confidence is a model self-report, not a measured probability. Queries and document text are treated as data; malicious instructions in evidence must not override the system rubric.

Judge consistency review identified {len(inconsistent_judgments)} fixed-refusal cases with faithfulness below 5, contrary to the rubric's abstention rule. Raw judgments are preserved without silently correcting or cherry-picking scores. See [judge_consistency_review.csv](evaluation/{manifest['run_id']}/judge_consistency_review.csv). These disagreements weaken interpretation of the aggregate scores and require independent adjudication before model selection or deployment. The curriculum-style question set overlaps prior exercise topics and is not a held-out generalization benchmark.

## Hallucination analysis

There are {len(hallucinations)} scored faithfulness≤2 cases. See [case analysis](evaluation/{manifest['run_id']}/hallucination_analysis.md), [full flagged records](evaluation/{manifest['run_id']}/hallucination_cases.json), and each case checkpoint for answer, evidence and trace. Confirm the judge's allegation in the original PDF before assigning a definitive cause. Improve retrieval for missing banks/periods, prohibit unsupported novelty claims, and calculate changes only from cited matching inputs. Human review template: [human_review.csv](evaluation/{manifest['run_id']}/human_review.csv).

## Cost and latency

Per-query operating cost = query embedding API cost + assumed retrieval service cost + generation/tool-planning API cost. Local embeddings have zero API cost; CPU/electricity are excluded. Retrieval runs locally, so its service charge is explicitly assumed US$0/query; substitute a deployment budget before production costing. Multi-step agent costs include every model iteration. Judge costs are evaluation overhead and are reported separately. Actual input, cached input and output tokens are recorded in `reports/evaluation/api_usage.jsonl`. Prices are estimates checked against official model pages on {manifest['pricing_date']}: GPT-4o-mini US$0.15 input, US$0.075 cached input, US$0.60 output per million tokens; text-embedding-3-small US$0.02 per million. No automatic model substitution is performed. API retries or provider-side billing adjustments may not be fully reflected by returned usage.

Measured operating API cost across generated cases: US${totals['operating_usd']:.6f}; judge overhead: US${totals['judge_usd']:.6f}. Annual projection = mean operating cost ×1,000×365 (365,000 queries); it excludes one-time document embedding, reference-packet setup, hosting, analyst time and monitoring. Latency is observed end-to-end generation time excluding model load, judge time and cached report loading. Up to {manifest['workers']} cases run concurrently, so timings reflect that workload and are not a controlled production SLA. No empirical analyst-time savings are claimed.

## Experiment tracking and reproduction

MLflow experiment `canadian_bank_rag_step6` uses `models/mlflow.db`. It logs one run per complete configuration with corpus/test fingerprints, prompt/rubric artifacts, traces, quality/cost/latency metrics and case outputs. [MLflow run IDs](evaluation/{manifest['run_id']}/mlflow_runs.json) connect the comparison to tracked experiments. Run notebook 6 or `python -m src.run_evaluation --live`; completed checkpoints are reused. Changing code, corpus or test questions creates a new run identity. All three configurations must complete before claiming the step 6 comparison is complete.

Sources: [OpenAI model pricing](https://developers.openai.com/api/docs/models/gpt-4o-mini), [embedding pricing](https://developers.openai.com/api/docs/models/text-embedding-3-small), [structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs), [MLflow tracking](https://mlflow.org/docs/latest/ml/tracking/tracking-api/).
'''
    (reports/'evaluation_report.md').write_text(evaluation,encoding='utf-8')
    documentation = f'''# System documentation

This project supports first-pass research over public filings from Canada's Big Five banks. It provides evidence-backed drafts, not automated compliance decisions. Current evaluation: `{manifest['run_id']}`; recommendation: iterate.

## Architecture and pipeline

```mermaid
flowchart LR
  IR[Official bank PDF sources] --> ING[Notebook 1 parsing and chunking]
  ING --> META[Chunk JSONL and source metadata]
  META --> EMB[Notebook 2 embeddings]
  EMB --> DB[Persistent ChromaDB]
  Q[Analyst question] --> RET[Top-k or MMR retrieval]
  DB --> RET
  RET --> GEN[Grounded prompt or tool-calling agent]
  GEN --> ANS[Answer and citations]
  ANS --> JUDGE[Evidence-grounded LLM judge]
  JUDGE --> TRACK[Notebook 6 reports and MLflow]
```

Notebook 1 downloads maintained official PDF URLs, validates the content and records a manifest. It extracts PDF blocks, removes recognizable contents pages and recurring margin text, retains section/page metadata and creates token chunks with overlap. Heuristics cannot guarantee accurate multicolumn reading order or table structure; inspect flagged text. Notebook 2 stores local MiniLM or OpenAI embeddings with scalar bank/year/type/quarter/section metadata in Chroma. Current evaluation uses `{manifest['embedding_provider']}` embeddings and {len(runtime.corpus):,} indexed chunks across {len(runtime.filings)} filings. Keep the embedding preprocessing synchronized with its stored index; do not silently query a collection with a different model.

## Retrieval and prompts

Notebook 3 provides grounded answers with source labels. Notebook 4 compares basic, few-shot and evidence-checking prompts on 15 questions, validates structured source metadata and explores rewriting and edge cases. Prompt comparisons require human or explicitly automated ratings; a saved answer alone does not establish correctness. Step 6 uses exact step 4 prompt constants and fixed retrieval scopes for fair comparison. Structured output requires answer, sources (bank/document/section/year/label), confidence and caveats, with metadata checked against retrieved records.

Notebook 5 defines search_filings, get_filing_list and compare_sections with function schemas. Bank code BNS is canonical for Scotiabank; the tool accepts the name as an alias. It checks available filings, records tool arguments/observations and public action summaries, caps model iterations, and reports cost. An unavailable old filing prohibits asserting a risk is newly introduced. Source text can be incomplete; an absence in search results is not proof of absence in a report. The agent's self-directed searches may still miss relevant periods, and these failures must be evaluated rather than hidden.

## Evaluation and monitoring

Notebook 6 runs a 30-question set across three configurations and logs real usage and judge scores. See [evaluation_report.md](evaluation_report.md) for exact metrics, rubric, costs and limitations. Thresholds are faithfulness≥3.5 and relevance≥4.0; sample failures and missing manual reviews prevent a production recommendation even if an average passes. Inspect faithfulness≤2 cases, judge/rubric inconsistencies, unknown citations, missing years, tool errors, token spikes and incomplete answers. Check latency/cost distributions and compare production question mix with the test set. Add independently adjudicated ground truth and an analyst task-time study before claiming business savings.

## Operations and data handling

Use a local `.env` for OPENAI_API_KEY; never commit or print the value. Only public filing text is sent to the generation/judge API. Keep `data/raw`, processed source metadata, embedding settings and Chroma together for reproducibility. Case checkpoints, corpus/test/code hashes and raw usage enable audit and resumable runs. Generation and judge costs are separate. The API budget cap is estimated, not a billing guarantee. Install full MLflow to support the SQLite store and local UI. Do not rerun paid calls merely to display already-computed results.

Known data risks include incomplete BMO quarterly coverage, missing statutory filing dates, conservative English filtering and extracted tables. Stored section names are heuristic, not LLM-authored ground truth. Final human reviews and a measured time-savings study remain outstanding.
'''
    (reports/'system_documentation.md').write_text(documentation,encoding='utf-8')
