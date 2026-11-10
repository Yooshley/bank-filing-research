# Evaluation report

Evaluation identifier: `54a68bec798eab0e`. Status: **complete**, 90/90 answers judged. Recommendation: **iterate**.

## Method and test coverage

Thirty fixed questions cover all five banks, credit/market/operational/liquidity/climate/model risk, and factual (15), comparative (5), trend (5) and edge-case (5) requests. See [test set](evaluation/54a68bec798eab0e/test_set.json). Every configuration receives identical questions and bank/year requirements. The agent chooses its own tools; baseline/MMR retrieval reserves evidence for requested scopes rather than allowing one bank to consume all result slots. Each non-agent question has an eight-chunk total budget; MMR selects from up to 30 candidates per scope with lambda 0.7. Those settings are logged as parameters.

The three configurations are basic top-k with the step 4 basic prompt, MMR with schema-constrained answer/sources/confidence/caveats, and the step 5 tool-calling agent. The shared notebook definitions are loaded without executing install/example cells. The index and embedding preprocessing from notebook 2 are retained. Corpus fingerprint: `a6dcfee74676d1730446760df577122fdf02d97b3f6ca5f15b703ceb2c5d618a`. Test fingerprint: `28f1d5c72f1f7dfc8045543af7c32d5b0efb129f16058de18191d3ae72474e2d`.

This run reuses 58 unaffected completed baseline/MMR records from `1622ffdffd9cfdf4` after confirming identical corpus/test fingerprints and exact source metadata. All changed agent cases and previously rejected structured cases were generated and judged again. Each reused case records its origin_case_id and reused_from_run; the original API ledger remains the cost evidence. See reuse_provenance.json for the explicit lineage.

## Master comparison

- basic_top_k: 30/30 scored; faithfulness 4.53/5, relevance 4.60/5, completeness 4.33/5; 1 grounding-risk cases; mean latency 5.19s (p95 9.70s); operating cost US$0.000748/query; annual projection US$272.97; quality targets met.
- mmr_structured: 30/30 scored; faithfulness 4.57/5, relevance 4.67/5, completeness 4.43/5; 1 grounding-risk cases; mean latency 6.49s (p95 12.01s); operating cost US$0.000768/query; annual projection US$280.49; quality targets met.
- agentic: 30/30 scored; faithfulness 4.57/5, relevance 4.63/5, completeness 4.33/5; 1 grounding-risk cases; mean latency 15.56s (p95 37.95s); operating cost US$0.001503/query; annual projection US$548.74; quality targets met.

The complete machine-readable comparison is [master_comparison.csv](evaluation/54a68bec798eab0e/master_comparison.csv); individual scores are [per_question_metrics.csv](evaluation/54a68bec798eab0e/per_question_metrics.csv). Quality targets from the curriculum are faithfulness ≥3.5/5 and relevance ≥4.0/5. A configuration is marked complete only after all 30 answers are generated and judged. Errors remain visible and are not converted to zero scores or discarded without a count.

## Judge and grounding limitations

The LLM judge scores faithfulness, relevance and completeness on 1–5 integer rubrics with reasons and unsupported-claim lists. Faithfulness is checked against the exact evidence available to each answer; completeness additionally uses the same fixed reference packet for all configurations. Appropriate abstentions can be faithful but incomplete when reference evidence supports an answer. Unknown-bank/future/ambiguous questions permit a complete refusal or clarification. The judge is blinded to configuration names. Both generator and judge are `gpt-4o-mini`; self-evaluation bias remains. Reference packets are retrieved evidence, not expert-written gold answers or exhaustive whole-document review. Human ratings are preserved separately and remain required for assurance.

Citation checks validate source identifiers and structured metadata; they do not prove that every claim is supported. Confidence is a model self-report, not a measured probability. Queries and document text are treated as data; malicious instructions in evidence must not override the system rubric.

Judge consistency review identified 3 fixed-refusal cases with faithfulness below 5, contrary to the rubric's abstention rule. Raw judgments are preserved without silently correcting or cherry-picking scores. See [judge_consistency_review.csv](evaluation/54a68bec798eab0e/judge_consistency_review.csv). These disagreements weaken interpretation of the aggregate scores and require independent adjudication before model selection or deployment. The curriculum-style question set overlaps prior exercise topics and is not a held-out generalization benchmark.

## Hallucination analysis

There are 3 scored faithfulness≤2 cases. See [case analysis](evaluation/54a68bec798eab0e/hallucination_analysis.md), [full flagged records](evaluation/54a68bec798eab0e/hallucination_cases.json), and each case checkpoint for answer, evidence and trace. Confirm the judge's allegation in the original PDF before assigning a definitive cause. Improve retrieval for missing banks/periods, prohibit unsupported novelty claims, and calculate changes only from cited matching inputs. Human review template: [human_review.csv](evaluation/54a68bec798eab0e/human_review.csv).

## Cost and latency

Per-query operating cost = query embedding API cost + assumed retrieval service cost + generation/tool-planning API cost. Local embeddings have zero API cost; CPU/electricity are excluded. Retrieval runs locally, so its service charge is explicitly assumed US$0/query; substitute a deployment budget before production costing. Multi-step agent costs include every model iteration. Judge costs are evaluation overhead and are reported separately. Actual input, cached input and output tokens are recorded in `reports/evaluation/api_usage.jsonl`. Prices are estimates checked against official model pages on 2026-10-27: GPT-4o-mini US$0.15 input, US$0.075 cached input, US$0.60 output per million tokens; text-embedding-3-small US$0.02 per million. No automatic model substitution is performed. API retries or provider-side billing adjustments may not be fully reflected by returned usage.

Measured operating API cost across generated cases: US$0.090592; judge overhead: US$0.159154. Annual projection = mean operating cost ×1,000×365 (365,000 queries); it excludes one-time document embedding, reference-packet setup, hosting, analyst time and monitoring. Latency is observed end-to-end generation time excluding model load, judge time and cached report loading. Up to 3 cases run concurrently, so timings reflect that workload and are not a controlled production SLA. No empirical analyst-time savings are claimed.

## Experiment tracking and reproduction

MLflow experiment `canadian_bank_rag_step6` uses `models/mlflow.db`. It logs one run per complete configuration with corpus/test fingerprints, prompt/rubric artifacts, traces, quality/cost/latency metrics and case outputs. [MLflow run IDs](evaluation/54a68bec798eab0e/mlflow_runs.json) connect the comparison to tracked experiments. Run notebook 6 or `python -m src.run_evaluation --live`; completed checkpoints are reused. Changing code, corpus or test questions creates a new run identity. All three configurations must complete before claiming the step 6 comparison is complete.

Sources: [OpenAI model pricing](https://developers.openai.com/api/docs/models/gpt-4o-mini), [embedding pricing](https://developers.openai.com/api/docs/models/text-embedding-3-small), [structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs), [MLflow tracking](https://mlflow.org/docs/latest/ml/tracking/tracking-api/).
