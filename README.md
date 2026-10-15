# Diabetes Awareness — Hackathon Prototype

Streamlit application with:
- Email/password registration and login (SQLite + Argon2 password hashes)
- Existing trained HistGradientBoosting model integration
- Educational assessment form
- Global permutation feature importance
- Model metrics and dataset analytics
- Rule-based health FAQ chatbot

## 1. Place existing files
Copy your trained artifacts into `models/`:
- `diabetes_model.joblib`
- `feature_names.joblib`
- `histgradientboosting_results.csv`

Copy `diabetes_health_indicators.csv` into `data/`.

This ZIP intentionally does not include your large dataset or your trained model binary. Keep your existing files and copy the app files into the same project folder, or put the CSV/model artifacts into this folder.

## 2. Install
From this folder in the VS Code terminal:
```powershell
python -m pip install -r requirements.txt
```

## 3. Run
```powershell
python -m streamlit run app.py
```
Open the local address shown in the terminal, usually http://localhost:8501.

## Notes
- The form uses the same 13 feature names and the dataset's coded age-group and general-health values.
- The dataset represents existing diabetes-related status in a US survey dataset. This prototype is NOT a clinically validated predictor of future diabetes onset and is not a diagnostic tool.
- The login database is local demo storage. Do not deploy with real sensitive health data without security, privacy, access-control, and compliance review.
- If you change the trained model's feature set, update the form to match it exactly.
