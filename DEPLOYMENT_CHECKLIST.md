# ForgeFlex release & deployment checklist

## Local release gate
- [ ] `python -m pytest -q` passes
- [ ] `python -m forgeflex.cli --help` includes all expected commands
- [ ] `streamlit run app.py` opens locally
- [ ] Dashboard can handle missing artifacts gracefully
- [ ] Generate fresh scenario outputs and inspect JSON/CSV
- [ ] No API keys, private data, caches, or machine-specific absolute paths committed

## GitHub
- [ ] Review source and licenses for the HIPE dataset
- [ ] Do not commit raw dataset archives or generated `outputs/` unless deliberately curated
- [ ] Add README_FINAL.md as README.md after reviewing it personally
- [ ] Push to a repository owned/controlled by the participant

## Streamlit Community Cloud
- [ ] Choose repository and branch
- [ ] Main file path: `app.py`
- [ ] Python version supported by cloud build image; `runtime.txt` currently requests Python 3.11
- [ ] Confirm dependencies install from `requirements.txt`
- [ ] Test cloud deployment from a clean state

## Submission integrity
- [ ] Verify hackathon terms and restrictions on unauthorized AI-generated content
- [ ] Personally inspect, understand, test, and adapt all code and claims before submission
- [ ] Clearly distinguish actual observations from simulations and assumptions
