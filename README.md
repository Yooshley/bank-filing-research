# Canadian Bank Filing Research

Canadian bank filing ingestion and vector retrieval notebooks.

## Setup

## Local conversation demo

For remote presentation access, see [Community Cloud deployment](docs/community_cloud.md).
The app supports Streamlit secrets and a checksum-verified portable index bundle.

From the project root, run `python -m streamlit run app.py --server.address 127.0.0.1`.
Open the localhost URL printed by Streamlit. The app uses the existing Chroma
index and project `.env`; run notebooks 1 and 2 first if the index is missing.
Install dependencies with `python -m pip install -r requirements.txt`.

The app first uses a structured LLM interpreter to combine the current message,
recent clarification dialogue and pending research request. Python validates
bank aliases, explicit bank/year mentions, corpus years and report types before
retrieval. New questions replace the topic; short answers such as `2025` or
`BMO` fill missing fields without losing the original request. Ambiguous
references ask for clarification. Unknown bank names cannot be mapped to a
supported alias. Interpretation can still fail, and errors stop retrieval.
Expand multi-turn tests before claiming broad conversational reliability.
The interpreter and Research assistant use `gpt-6-astra` with low reasoning effort.
This upgrades the app's answer model from the benchmark's `gpt-4o-mini`.
Run offline checks with
`python -m unittest discover -s tests -q`. The optional
`python demos/02_check_conversations.py` check makes paid interpretation calls
and tests 13 turns, including year/bank clarifications, comparisons, topic
changes and unsupported banks. Its results stay under ignored `.local/`.
The paid `python demos/03_check_research.py` check also runs complete answers
against the local index: BMO 2024 credit risk, switching to TD, asking "its"
liquidity coverage ratio, and comparing that ratio with BMO. Use `--data-root`
and `--env-file` when the index or environment file lives in another checkout.

Research assistant with generated answers is on initially. It uses the existing API key and
has an estimated US$5 ledger budget per browser session, excluding any embedding
API usage not covered by reservations. Local MiniLM embeddings avoid those API
charges. Evidence-only mode still requires an API key for conversation
interpretation, although it skips answer generation. Interpretation charges
are included in displayed turn costs, with a separate estimated US$1
interpretation budget per conversation. The limits are not billing guarantees.
Research assistant combines BM25-style keyword search with local vector search
inside each validated bank/year/report scope. Reciprocal-rank fusion and reserved
results for each subquery preserve evidence for specific metrics and comparisons.
The interpreter proposes up to three topic searches; the research workflow can
make one additional scoped search when its draft lacks supported evidence.
The answer model drafts short findings with quotations. A separate evidence-review
call removes unsupported clauses, corrects citations, and checks metric units and
periods. Python checks quotation spans, numeric references, scope and source labels
before display. Metric answers show one point; all answers are capped at 160 words.
These checks reduce errors but do not establish ground-truth financial accuracy.
Original passages and filing links remain available for manual verification.
The optional Basic top-k baseline retains the historical small-model behavior
for comparison. The original notebook agent and benchmark methods remain in the notebooks.
The pending request updates during clarification, before retrieval. Inspect the
`Understood research question` panel to see the complete request used for
retrieval. The default research workflow expands searches automatically within
bounded limits. Use New conversation to clear
the discussion. Live answers use fresh filing retrieval; previous generated
financial answers are never treated as evidence. These changes add API calls
and are not included in the historical benchmark scores or cost averages.

The historical evaluation scores do not evaluate this new presentation pipeline.
The stronger model and evidence-review call cost more than the original demo.
Token estimates use model-specific input/cached-input/output rates; turn latency
includes interpretation. Session logs and draft/review diagnostics stay under ignored `.local/demo`.
The four-turn local smoke check on 2026-11-11 returned cited BMO/TD credit-risk
summaries, TD's 138% quarterly average LCR, and the TD 138% versus BMO 132%
comparison. Those turns took approximately 9–32 seconds after model startup,
with a mean estimated full-turn API cost of US$0.254. This small check is not
a new benchmark or a guarantee for other questions.

## Notebook setup

Use Python 3.11 and create a virtual environment at the project root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
jupyter lab
```

Set `OPENAI_API_KEY` in the local `.env` before running generation or judging.
Local embeddings do not require an OpenAI key. Run notebooks 1 through 6 in
order. The saved corpus covers fiscal 2024 and 2025; later filings require
updating the source catalog.

Downloaded PDFs, processed text, model caches, Chroma and the local MLflow
database are excluded from Git. Rebuild them with notebooks 1 and 2. Saved
evaluation reports and case records remain available for inspecting results;
the local MLflow UI requires rebuilding its database through a live evaluation.
Study documents and local notebook backups are excluded from publication.

## Run the data pipeline

Open `notebooks/01_data_ingestion.ipynb` and run its cells in order. The setup
cell installs dependencies. The first data cell automatically calls
`demos/01_download_filings.py` to download real English annual reports and AIFs
for RBC, TD, BMO, Scotiabank and CIBC, plus verified quarterly reports. It creates the
source manifest, then the notebook parses, cleans, chunks and audits the PDFs.

You can also acquire the filings from a terminal at this project root:

```powershell
python -m pip install requests pymupdf
python demos/01_download_filings.py
```

Download only annual reports and AIFs, or a subset of banks/years:

```powershell
python demos/01_download_filings.py --annual-only
python demos/01_download_filings.py --banks RBC TD --years 2024 2025
```

The maintained catalog `data/filing_sources.json` contains official bank PDF
URLs for fiscal 2024 and 2025. Default years are the latest two completed
October 31 fiscal years; adding a later fiscal year requires updating verified
catalog links. The downloader validates PDF content, bank, year and filing type,
reuses valid local files and records failures rather than inventing sample data.
The current catalog contains 44 reports: all ten annual reports, all ten AIFs,
and Q1–Q3 reports for RBC, TD, Scotiabank and CIBC across both years. BMO's six
Q1–Q3 reports are still missing; notebook 1 reports that gap explicitly.
It does not automate SEDAR+ search. It follows the instructor demo's structure
using the direct investor-relations download alternative described in Module 4.

## Files and provenance

- `data/raw/`: downloaded PDFs.
- `data/filing_manifest.csv`: source URLs and retrieval metadata used by notebook 1.
- `data/download_report.json`: downloaded/cached status, SHA-256, page counts,
  retrieval timestamp and errors.
- `data/processed/`: cleaned text/page records, filing index, chunks, charts and audits.
- `notebooks/02_embedding_vectordb.ipynb`: local embeddings, persistent Chroma,
  optional OpenAI comparison, similarity/MMR retrieval and manual precision@3.
- `notebooks/03_rag_pipeline.ipynb`: grounded RAG and 15 question tests.
- `notebooks/04_prompt_optimization.ipynb`: prompt variants, structured sources,
  rewriting and edge cases.
- `notebooks/05_agentic_workflow.ipynb`: three filing tools and multi-step traces.
- `notebooks/06_evaluation.ipynb`: 30 questions across three configurations,
  LLM-as-judge scores, hallucination analysis, cost/latency and MLflow.

Annual reports include year-end/Q4 reviews and are registered once. Their
`covers_q4` field lets notebook 1 distinguish that coverage from a missing Q4
filing. Statutory `filing_date` values remain blank unless independently known;
download time and HTTP modification time are not used as filing dates. Known
dates and additional manifest entries are preserved on rerun.

Network access is required for the first download and tokenizer/model setup.
After acquisition, set `DOWNLOAD_MISSING=False` in notebook 1 to process local
files. Review `download_report.json` and notebook 1 coverage/extraction diagnostics
before running notebook 2. Notebook 2 requires manual relevance judgments to
report retrieval precision; it does not fabricate scores.

## Evaluate and reproduce the results

Run notebooks 1 through 5 in order. The embedding provider must match the saved
Chroma collection and the preprocessing in notebook 2. For answer generation
and live judging, create a project `.env` containing `OPENAI_API_KEY=...`.
The key must not be committed or printed. Notebook-local `.env` files also work.

Install the evaluation dependencies in the same environment as your notebook:

```powershell
python -m pip install "chromadb>=1,<2" sentence-transformers openai python-dotenv pandas numpy matplotlib tiktoken "mlflow>=3,<4"
```

Open notebook 6. Its default loads the latest saved evaluation without making
generation or judge calls. To generate missing cases, set
`RUN_LIVE_EVALUATION=True`, or run from this project root:

```powershell
python -m src.run_evaluation --live --workers 3 --budget-usd 5
```

The estimated US$5 cap applies to the cumulative evaluation API ledger; it is
not a billing guarantee. Query operating costs and one-time judge costs are
separate. Checkpoints save after generation and judging, so rerunning identical
inputs reuses completed cases. Changed corpus, source code, models or questions
produce a new run identifier. Do not delete checkpoints to refresh a report.

`reports/evaluation/latest_run.json` points to the current run. Each run contains
the test set, reference packets, 90 answer/judge checkpoints, per-question metrics,
the master comparison, hallucination analysis, human-review sheet, charts and
MLflow run IDs. The top-level `reports/` folder contains
`executive_summary.md`, `system_documentation.md`, `evaluation_report.md` and
`step_4_5_verification.md`. These reports disclose incomplete runs and corpus
gaps; automated judging is never represented as completed human review.

MLflow stores three complete configuration runs in `models/mlflow.db` under
experiment `canadian_bank_rag_step6`. For the local UI, install full `mlflow`
and run:

```powershell
python -m pip install "mlflow>=3,<4"
mlflow ui --backend-store-uri sqlite:///models/mlflow.db --port 5000
```

Then open [MLflow](http://localhost:5000). A live UI is not required for the
notebook's saved comparison. Validate the offline evaluation protocol tests with:

```powershell
python -m unittest discover -s tests -p test_evaluation.py -v
```

Run all offline tests from the project root with
`python -m unittest discover -s tests -v`. Tests do not call the model APIs.

Fill the existing prompt/agent review CSVs after checking answers against the
actual filing text. The evaluation run creates a separate `human_review.csv`
and preserves any ratings on report refresh. LLM scores are evidence-based
screening results, not independent gold-standard financial accuracy. The current
question set follows curriculum examples and overlaps prior question topics;
it is not a held-out generalization benchmark. Analyst time savings still need a
timed human study.

Official sources: [RBC financial reports](https://www.rbc.com/investor-relations/financial-information.html),
[TD financial reports](https://www.td.com/ca/en/about-td/for-investors/investor-relations/financial-information/financial-reports/annual-reports),
[BMO annual filings](https://www.bmo.com/main/about-bmo/investor-relations/annual-reports-proxy-circulars),
[Scotiabank financial results](https://www.scotiabank.com/ca/en/about/investors-shareholders/financial-result.html),
[CIBC annual filings](https://www.cibc.com/en/about-cibc/investor-relations/annual-reports-and-proxy-circulars.html).
