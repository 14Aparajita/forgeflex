# ForgeFlex Stage 4 — Interactive Judge Demo

This package adds an optional local Streamlit interface. It does not replace the existing CLI and does not overwrite project modules other than adding `forgeflex/dashboard_utils.py`.

## Install
Run from the project root in the existing `tfenv` environment:

```powershell
Expand-Archive -Path "$HOME\Downloads\forgeflex_stage4_interactive_v1.zip" -DestinationPath . -Force
python -m pip install -r .\requirements-demo.txt
python -m pytest -q
```

The archive adds `app.py`, `forgeflex/dashboard_utils.py`, `tests/test_dashboard_utils.py`, `requirements-demo.txt`, and this guide. It does not replace `cli.py` or alter existing data/output files.

## Launch
From the project root:

```powershell
python -m streamlit run .\app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`). Open it in your browser. Keep the terminal running while demonstrating the app.

## What the demo provides
- Browse the aligned public HIPE power series.
- Inspect sensor validation recommendations and saved walk-forward metrics.
- Adjust synthetic job count, seed, scheduling horizon, and slot duration.
- Run the existing baseline and tariff-aware scheduling comparison.
- Inspect schedules, summary metrics, and scheduled aggregate-demand profiles.

## Data provenance and limitations
- HIPE is a public historical dataset, not the user's factory telemetry.
- Sensor-placement results are retrospective heuristic priorities for validation, not automatic installation advice.
- The scheduling sandbox uses synthetic jobs and an illustrative tariff unless real CSV inputs are provided to the existing CLI; this dashboard's interactive controls generate synthetic jobs.
- A lower simulated tariff bill is not necessarily lower energy consumption. Do not claim factory savings, payback, or ROI from this simulation.
- This is a local demo and does not connect to PLCs, meters, ERP systems, or live factory equipment.

## Troubleshooting
- If the sidebar paths do not exist, edit them to match your local files. The app reports unavailable sources rather than fabricating values.
- If Streamlit is not found, run `python -m pip install -r .\requirements-demo.txt` in the activated environment.
- If imports fail, verify the terminal's working directory is the project root containing `forgeflex` and `outputs`.
