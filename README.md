# CKD Progression Prediction using LSTM + Temporal Instability Index

> **B.Tech CSE (AIML) Minor Project** — Predicting Chronic Kidney Disease progression from longitudinal EHR data using deep learning with a novel Temporal Instability Index (TII).

---

## 🔬 Research Contribution

This project introduces a **3-way model comparison** for CKD progression prediction:

| Model | Description | Key Insight |
|-------|-------------|-------------|
| **Baseline (RF)** | Random Forest on last-visit features only | Current clinical practice approximation |
| **LSTM** | Bidirectional LSTM on full patient time-series | Captures temporal trends |
| **LSTM + TII** ★ | LSTM enhanced with Temporal Instability Index | **Proposed method** — captures biomarker volatility |

**TII = Standard Deviation / Mean** (Coefficient of Variation) computed per patient across all visits for eGFR, Creatinine, and Systolic BP. Patients with unstable biomarkers (high TII) are at greater risk of progression.

---

## 📊 Dataset

- **Source**: [MIMIC-IV v3.1](https://physionet.org/content/mimiciv/3.1/) (real-world ICU data)
- **Extraction**: `src/extract_mimic.py` processes ~160M lab records through chunked loading
- **Size**: ~240,000 longitudinal visits across ~40,000 patients
- **Clinical Features**: eGFR (CKD-EPI 2021), Creatinine, HbA1c, Hemoglobin, Blood Pressure, Age, Gender
- **CKD Staging**: KDIGO 2012 guidelines
- **Dynamic Sequences**: Adapts to each patient's full visit history (max 116 visits)

---

## 🏗️ Project Structure

```
ckd_project/
├── src/
│   ├── extract_mimic.py              # MIMIC-IV data extraction
│   ├── preprocess.py                 # Feature engineering + TII computation
│   ├── eda.py                        # Exploratory data analysis
│   ├── baseline_model.py             # Model 1: RF/LR/GBM baselines
│   ├── lstm_model.py                 # Models 2 & 3: LSTM and LSTM+TII
│   └── comparison_and_explainer.py   # 3-way comparison + clinical explanations
├── backend/
│   └── app.py                        # Flask REST API (serves all 3 models)
├── frontend/
│   └── dashboard.html                # Clinical dashboard
├── data/                             # Generated datasets (gitignored)
├── models/                           # Trained models (gitignored)
├── plots/                            # Evaluation plots
├── results/                          # Metrics CSVs and explanations
└── requirements.txt
```

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Step-by-Step Execution

1. **Place MIMIC-IV files** in your Downloads folder:
   - `patients.csv.gz`, `admissions.csv.gz`, `diagnoses_icd.csv.gz`
   - `omr.csv.gz`, `labevents.csv.gz`, `d_labitems.csv.gz`

2. **Extract the dataset:**
   ```bash
   python src/extract_mimic.py
   ```

3. **Preprocess and compute TII features:**
   ```bash
   python src/preprocess.py
   ```

4. **Run EDA:**
   ```bash
   python src/eda.py
   ```

5. **Train baseline models:**
   ```bash
   python src/baseline_model.py
   ```

6. **Train LSTM and LSTM+TII models:**
   ```bash
   python src/lstm_model.py
   ```

7. **Compare models and generate explanations:**
   ```bash
   python src/comparison_and_explainer.py
   ```

8. **Launch dashboard (optional):**
   ```bash
   python backend/app.py
   # Visit http://localhost:5000
   ```

---

## 🧠 Key Technical Details

### TII Computation
```
TII_egfr       = std(egfr across visits)       / mean(egfr across visits)
TII_creatinine = std(creatinine across visits)  / mean(creatinine across visits)
TII_systolic_bp = std(systolic_bp across visits) / mean(systolic_bp across visits)
```

### LSTM Architecture
```
Input → Masking(0.0) → BiLSTM(64) → Dropout(0.3) → LSTM(32) → Dropout(0.2) → Dense(16, ReLU, L2) → Sigmoid
```

### Dynamic Sequence Padding
- Sequences adapt to the longest patient history in the dataset
- Left-zero-padding with Masking layer for shorter histories
- No artificial truncation — full clinical history is preserved

---

## 📈 Outputs

| Output | Path |
|--------|------|
| Extracted dataset | `data/ckd_longitudinal.csv` |
| Preprocessed data | `data/ckd_preprocessed.csv` |
| LSTM sequences | `data/ckd_sequences.npz` |
| LSTM+TII sequences | `data/ckd_sequences_tii.npz` |
| Baseline model | `models/baseline_rf_model.pkl` |
| LSTM model | `models/lstm_model.h5` |
| LSTM+TII model | `models/lstm_tii_model.h5` |
| Comparison plots | `plots/comparison_*.png` |
| Metrics table | `results/model_comparison.csv` |
| Clinical explanations | `results/example_explanations.json` |

---

## 📚 References

- **MIMIC-IV**: Johnson et al., MIMIC-IV (version 3.1). PhysioNet, 2024.
- **CKD-EPI**: Inker et al. New Creatinine- and Cystatin C-Based Equations. NEJM, 2021.
- **KDIGO**: KDIGO 2012 Clinical Practice Guideline for CKD.
- **TII Concept**: Coefficient of Variation as a measure of biomarker instability.
