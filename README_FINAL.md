# ForgeFlex — Adaptive-Sensing Production Energy Orchestrator

**Track:** Smart Manufacturing — Industrial Energy & Process Efficiency

ForgeFlex is a prototype decision-support workflow for SME manufacturing. It consolidates public historical power analytics, aggregate-only machine-power inference validation, retrospective submeter-validation priorities, and tariff-aware production scheduling simulations. It is not a live industrial control system.

## Evidence boundaries

- HIPE is a public historical one-week dataset; it is not the user's factory.
- Activity is inferred from a 0.05 kW threshold, not official machine-state ground truth.
- Sensor-placement rankings are heuristic validation priorities, not ROI or causal sensor-value estimates.
- Default jobs and tariffs are synthetic/illustrative. Simulated cost shifts are not measured energy or cost savings.
- The current greedy and conservative schedulers are heuristics, not globally optimal solvers. Product quality is not modeled.

## Quick start (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pytest -q
python -m forgeflex.cli --help
python -m forgeflex.cli schedule --output-dir outputs/scheduling_demo --seed 42 --job-count 20 --horizon-slots 24 --slot-hours 1.0
python -m forgeflex.cli experiments --output-dir outputs/scheduling_experiments --seeds 42 7 21 --job-counts 10 20 30 --horizon-slots 24 --slot-hours 1.0
python -m forgeflex.cli constraint-aware-schedule --output-dir outputs/constraint_aware_demo --seed 42 --job-count 20 --horizon-slots 24 --slot-hours 1.0
```

## Dashboard

```powershell
streamlit run app.py
```

The dashboard reads local artifacts from `outputs/`. To show historical data, first run the HIPE preprocessing/audit, walk-forward validation, sensor-placement and decision-report commands with your local dataset/artifacts; paths can be configured in the sidebar. A fresh clone without local artifacts should be treated as an empty workspace until artifacts are generated.

## Current CLI commands

`baseline`, `hipe`, `activity-audit`, `infer`, `walkforward`, `two-stage`, `sensor-placement`, `decision-report`, `schedule`, `experiments`, `constraint-aware-schedule`. Use `python -m forgeflex.cli <command> --help` for flags.

## Deployment

### Streamlit Community Cloud
1. Create a GitHub repository you control and upload this release folder (excluding raw data, local outputs, credentials and backups).
2. Ensure `app.py`, `requirements.txt`, `forgeflex/`, and `tests/` are at repository root.
3. In Streamlit Community Cloud, deploy that repository with main file `app.py`.
4. The cloud app has no access to your Windows `outputs/` directory. Commit only small, permitted, non-sensitive demo artifacts if you want prepopulated charts, or have the dashboard generate synthetic examples itself. Never commit private factory data.
5. After deployment, test from a clean browser and verify the app's empty-artifact state.

### Docker

```bash
docker build -t forgeflex .
docker run --rm -p 8501:8501 forgeflex
```

## Final verification checklist

- Run all tests in the deployment environment.
- Run every CLI command used in the demo.
- Regenerate the report from known artifact paths.
- Demonstrate a flat tariff tie and at least one time-of-use scenario.
- Show deadline misses and unscheduled jobs, not just savings.
- Label every simulated output as simulation-only.
- Never claim factory ROI, energy savings, product quality preservation, or live control without real data and validation.

## Hackathon submission checklist

The official Smart Manufacturing challenge asks for a solution write-up, architecture/data-flow diagram, supporting design artifacts, quantified improvement against a defined baseline while preserving throughput and quality, and a deployment/business-model plan. The current synthetic simulation does **not** establish real-factory impact or quality preservation. The final presentation should state that limitation and define a pilot validation plan. Review official rules about original work and unauthorized AI-generated or plagiarized submissions before submitting: https://www.yuvayodhatech.com/terms
