# Streamlit Community Cloud

Deploy repository `yash9025/RegimeCast`, branch `main`, main file `dashboard/app.py`.

The dashboard directory contains its own `requirements.txt`. Streamlit checks the entrypoint directory before the repository root, so this bypasses automatic Poetry resolution and installs only the display dependencies. The source package is loaded from `src/`; no editable install is required. Python 3.13 is the recommended choice in Advanced settings for a new deployment, matching the development environment.

The supplied installation log describes a dependency resolver/Python-range error. It is separate from the missing-artifact issue, which would only occur after successful installation. Both are addressed here.

## Published data

When local `artifacts/catalog.json` is absent, the app uses `dashboard/snapshot/catalog.json`. All paths are portable and relative to that catalog. The bundle contains genuine saved forecasts, expert contributions, feature context and the latest 504 sessions of regime probabilities. Evaluation metrics and annual/uncertainty/error reports come from the already-published research reports.

No FRED key, GitHub token, raw price file, fitted model binary, database or model training is needed on Streamlit Cloud. The page identifies the data as a published research snapshot and displays the actual forecast as-of date. This is not a live price feed.

After this push, reboot the existing app from Manage app. Keep `dashboard/app.py` as the main file. If creating a new app, choose Python 3.13 in Advanced settings.

## Updating the published snapshot

Prepare new local data and forecasts first, then run:

```powershell
.\.venv\Scripts\python.exe tools/export_dashboard_snapshot.py
```

Publish the generated display files in a focused commit. If the underlying research data changes, regenerate and publish the matching evaluation reports as well; do not pair refreshed forecasts with unrelated research results.
