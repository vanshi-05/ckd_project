# Execution Guide — MIMIC-IV CKD Prediction Pipeline

## Pre-requisites

1. **MIMIC-IV files** in `C:\Users\Vanshika Selvam\Downloads\`:
   - `patients.csv.gz`, `admissions.csv.gz`, `diagnoses_icd.csv.gz`
   - `omr.csv.gz`, `labevents.csv.gz`, `d_labitems.csv.gz`

2. **Python packages**:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn joblib flask flask-cors scipy tensorflow
   ```

---

## Execution Order

Run from the `ckd_project` directory:

```bash
# Step 1: Extract MIMIC-IV data (~15–25 min for labevents)
python src/extract_mimic.py

# Step 2: Preprocess + compute TII features
python src/preprocess.py

# Step 3: Exploratory Data Analysis
python src/eda.py

# Step 4: Train baseline models (RF, LR, GBM)
python src/baseline_model.py

# Step 5: Train LSTM and LSTM+TII models
python src/lstm_model.py

# Step 6: 3-way comparison + clinical explanations
python src/comparison_and_explainer.py

# Step 7 (Optional): Launch dashboard
python backend/app.py
```

---

## What Each Step Produces

| Step | Script | Key Outputs |
|------|--------|-------------|
| 1 | `extract_mimic.py` | `data/ckd_longitudinal.csv` |
| 2 | `preprocess.py` | `data/ckd_sequences.npz`, `data/ckd_sequences_tii.npz`, `data/ckd_baseline.npz` |
| 3 | `eda.py` | `plots/eda_*.png` |
| 4 | `baseline_model.py` | `models/baseline_rf_model.pkl`, `results/baseline_results.csv` |
| 5 | `lstm_model.py` | `models/lstm_model.h5`, `models/lstm_tii_model.h5` |
| 6 | `comparison_and_explainer.py` | `results/model_comparison.csv`, `plots/comparison_*.png` |

---

## Troubleshooting

- **Memory errors during extraction**: The script uses 5M-row chunks. Close other apps.
- **TensorFlow not found**: `pip install tensorflow` (CPU version is fine for this project size)
- **Missing MIMIC files**: Ensure all 6 `.csv.gz` files are in your Downloads folder.
