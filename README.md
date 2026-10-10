# Sri Lanka Climate & Flood Risk Dashboard

A Flask frontend for exploring the modeled Sri Lanka district climate/flood-risk CSV.

## Project files
- `app.py` — Flask application and chart generation
- `templates/index.html` — responsive dashboard frontend
- `eda_config.pkl` — pickle containing dashboard column configuration only (not a trained ML model)
- `Dockerfile` — container build instructions
- `requirements.txt` — Python dependencies

## Run locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000 and upload your CSV.

## Run with Docker
Put `sri_lanka_flood_risk_modeled.csv` in a local `data` folder beside the Dockerfile, then:
```bash
docker build -t sri-lanka-flood-dashboard .
docker run --rm -p 5000:5000 -v "${PWD}/data:/app/data" sri-lanka-flood-dashboard
```
Open http://localhost:5000. On Windows PowerShell, `${PWD}` usually works; alternatively use the full path to your `data` folder.

The app also allows uploading a CSV in the browser. Uploaded files are stored in the container's `uploads` folder; mount a host folder to persist them:
```bash
docker run --rm -p 5000:5000 -v "${PWD}/data:/app/data" -v "${PWD}/uploads:/app/uploads" sri-lanka-flood-dashboard
```

## Important
The Kaggle path in a notebook is not automatically available inside Docker. Download the CSV from Kaggle and place it in `data/`, or upload it through the app. The pickle is configuration only; your EDA script does not train a prediction model, so a model pickle cannot be created from it without a separate model-training step.
