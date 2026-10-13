# ForgeFlex — Adaptive-Sensing Production Energy Orchestrator

**An energy-intelligence prototype for smarter industrial monitoring and production scheduling.**

ForgeFlex explores how industrial energy data can support better machine-level monitoring, energy forecasting, sensor-placement decisions, and production scheduling under time-varying electricity tariffs.

Built for the **Yuva Yodha Energy Tech Hackathon 2026 — Schneider Electric**, this software prototype combines data-driven analysis with an interactive dashboard to demonstrate an approach to industrial energy and process efficiency.

> **Prototype status:** ForgeFlex is a research and demonstration prototype, not a validated industrial control system. Results from public historical data and simulated scenarios must not be interpreted as verified savings at a real factory.

## The problem

Industrial facilities need to balance energy cost, equipment visibility, and production requirements. Installing additional meters everywhere can be expensive, while incomplete measurements make machine-level analysis difficult. Production scheduling can also affect electricity costs when tariffs vary over time.

ForgeFlex demonstrates a data-driven workflow for investigating these challenges using historical energy traces, predictive baselines, sensor-placement analysis, and scenario-based scheduling.

## What ForgeFlex does

- **Energy monitoring:** Processes and presents industrial energy data for analysis.
- **Energy forecasting:** Evaluates forecasting approaches against baseline methods using historical time-series data.
- **Adaptive-sensing analysis:** Ranks potential submetering candidates using retrospective validation evidence and explicit eligibility checks.
- **Energy-aware scheduling:** Explores production scheduling under time-of-use tariff scenarios.
- **Scenario comparison:** Reports evaluation metrics to help users understand the trade-offs between methods.
- **Interactive dashboard:** Provides a visual interface for exploring the prototype's available results.

The recommendations are decision-support outputs. They do not automatically install sensors, control machinery, or execute production schedules.

## Architecture

```text
Historical energy data
        |
        v
Data loading and preprocessing
        |
        v
Energy analysis and forecasting
        |
        +----------------------+
        |                      |
        v                      v
Sensor-placement analysis   Scheduling scenarios
        |                      |
        +-----------+----------+
                    |
                    v
          Metrics and evidence
                    |
                    v
          Streamlit dashboard
```

## Technology stack

- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn and supporting scientific Python libraries
- Pytest for automated tests

See `requirements.txt` for the project's dependency list and `docs/` for additional implementation and deployment documentation.

## Run locally

### 1. Clone the repository

```bash
git clone https://github.com/14Aparajita/forgeflex.git
cd forgeflex
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Start the dashboard

```bash
python -m streamlit run app.py
```

Open the local URL displayed by Streamlit in your browser.

## Run the tests

Run the project's automated tests with:

```bash
python -m pytest -q tests
```

If the parent directory's pytest configuration interferes with test discovery, the following command can be used in the development environment:

```powershell
python -m pytest -c NUL --import-mode=importlib -q .\tests
```

The development release has passed **41 automated tests** in the local environment. Test results should be rechecked after code or dependency changes.

## Data and reproducibility

ForgeFlex includes selected processed outputs and evaluation artifacts required for the demonstration. Large raw datasets are intentionally excluded from the Git repository.

The HIPE results use public historical energy data. They are not measurements from a Schneider Electric facility or from the user's factory. Before reproducing experiments, check the dataset's original source, licence, attribution requirements, and the project documentation.

Forecasting metrics, activity thresholds, sensor rankings, and scheduling outcomes depend on the data and evaluation assumptions. Sensor rankings are candidates for further investigation—not proof of return on investment. Scheduling results are scenario-based and must be checked against production constraints before any operational use.

## Deployment

The dashboard can be deployed using Streamlit Community Cloud:

1. Push the project to GitHub.
2. Sign in to Streamlit Community Cloud using GitHub.
3. Create a new app and select `14Aparajita/forgeflex`.
4. Select the `main` branch.
5. Set the main file path to `app.py`.
6. Deploy and inspect the application logs.
7. Verify the dashboard and all required artifacts in the deployed environment.

The cloud deployment must be tested separately from the local environment. Local test success does not guarantee that every dashboard feature or artifact will work in the cloud.

## Limitations and next steps

- Validate the workflow on representative industrial data.
- Test forecasting against appropriate temporal baselines and held-out periods.
- Validate candidate sensor placements with installation costs, electrical feasibility, and operational requirements.
- Evaluate scheduling against real machine constraints, deadlines, throughput, and product-quality requirements.
- Compare simulated energy costs with verified meter readings and actual tariff structures.
- Add monitoring, failure handling, and operational safeguards before considering production integration.

## Intended use

ForgeFlex is intended for hackathon demonstration, research, and decision-support experimentation. It is not a replacement for industrial energy-management systems, safety interlocks, or qualified engineering review.

## Repository

Source code: https://github.com/14Aparajita/forgeflex

Developed for the Yuva Yodha Energy Tech Hackathon 2026 by Schneider Electric.
