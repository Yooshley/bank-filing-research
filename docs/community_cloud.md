# Streamlit Community Cloud deployment

The app can be opened from another device using its deployed web URL. Deploy
the live index as well as the code; historical evaluation snippets are not a
replacement for the complete retrieval corpus.

1. Rebuild the missing local corpus with notebooks 1 and 2, using local embeddings.
2. Run `python -m src.cloud_index`. This exports the actual index to ignored
   `.local/cloud/filing-index.zip` and prints its SHA256 checksum.
3. Upload that archive as a GitHub release asset or to HTTPS object storage.
   Use a direct download URL. The archive contains public filing excerpts and
   embeddings; never include the project `.env` or local databases in it.
4. Publish this project to GitHub. In Streamlit Community Cloud, choose the
   repository and branch, set the entrypoint to `ml_3/app.py` if deploying from
   the parent repository, and select Python 3.11 in advanced settings.
5. Set these secrets in Streamlit's advanced settings, not in Git:

```toml
OPENAI_API_KEY = "your API key"
INDEX_BUNDLE_URL = "https://your-host/filing-index.zip"
INDEX_BUNDLE_SHA256 = "the 64 character checksum from the exporter"
```

The first real query downloads and verifies the archive, rebuilds Chroma and
downloads the MiniLM model. Allow extra time for the first run. Disk storage is
rebuildable; it must not be treated as durable. Keep the archive available.
Cloud secrets are loaded before backend initialization. Follow the sharing
settings to choose public access or invited viewers. Test the presentation URL
on the actual second device before the presentation and wake the app in advance.

For the current Research assistant update, keep the existing index bundle and
secrets: no index rebuild or notebook rerun is needed. After deployment choose
New conversation. The default pipeline uses `gpt-6-astra` for interpretation,
drafting and evidence review, so the API project must have access to that model.
Evidence Only still uses the API to interpret the conversation. Basic top-k is
an optional historical baseline; Research assistant is the recommended demo mode.

The local budget is per backend session and is not a global spending limit.
Use private sharing for the presentation and a separately budgeted API project.
No secrets should appear in the browser or source repository. This deployment
path has not been published until a real Community Cloud URL has been verified.

Official guidance: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app
