# Interpretable Bidirectional LSTM Modeling of CKD Progression from Multi-Visit Time-Series Records using Temporal Instability Index 

> **B.Tech CSE (AIML) Minor Project** — Predicting Chronic Kidney Disease progression from longitudinal EHR data using a BiLSTM deep learning model augmented with a novel Temporal Instability Index (TII), with global interpretability via SHAP and gradient saliency maps.

---

## 🔬 Research Contribution

This project introduces a **4-way model comparison** for CKD progression prediction:

| Model | Description | Key Insight |
|-------|-------------|-------------|
| **RF Baseline** | Random Forest on last-visit features only | Approximates current clinical practice |
| **BiLSTM** | Bidirectional LSTM on full patient time-series | Captures forward and backward temporal trends |
| **BiLSTM + TII** ★ | BiLSTM enhanced with Temporal Instability Index | **Proposed method** — captures named, auditable biomarker volatility |
| **Transformer Encoder (TE)** | Multi-head self-attention on visit sequences | Architecture-family comparison baseline |

**TII = Standard Deviation / Mean** (Coefficient of Variation) computed per patient across all visits for eGFR, Creatinine, and Systolic BP. Patients with unstable biomarkers (high TII) are at greater risk of progression. The TII values are appended to the final visit step of each patient's sequence before model input.

### Why BiLSTM+TII?

All three deep models significantly outperform the RF baseline (AUC ~0.795). BiLSTM+TII achieves AUC-ROC of 0.994 and F1 of 0.963 — statistically comparable to plain BiLSTM (AUC 0.998) and TE (AUC 0.995) — while being the **only model that provides clinically named, auditable risk signals** (eGFR-CV, creatinine-CV, SBP-CV) a nephrologist can directly inspect and challenge. It also achieves the **highest recall (0.973)** across all models, which is the most clinically critical metric for a disease monitoring setting.

---

## 📊 Dataset

- **Source**: [MIMIC-IV v3.1](https://physionet.org/content/mimiciv/3.1/) (de-identified critical care EHR, Beth Israel Deaconess Medical Center)
- **Cohort**: ~40,000 CKD patients (ICD-9: 585.x, ICD-10: N18.x), ~240,000 longitudinal visit records
- **Extraction**: `src/extract_mimic.py` processes ~160M lab records via chunked loading (5M rows/chunk)
- **Clinical Features**: eGFR (CKD-EPI 2021 equation), Creatinine, HbA1c, Hemoglobin, Systolic BP, Age, Gender
- **CKD Staging**: KDIGO 2012 five-stage classification based on eGFR thresholds
- **Sequence Length**: Up to 116 visits per patient; patients with fewer than 3 visits excluded
- **Progression Label**: Binary — stage worsened between consecutive visits (yes/no)

---

## 🏗️ Project Structure

```
ckd_project/
├── src/
│   ├── extract_mimic.py              # MIMIC-IV data extraction (chunked, ICD filtering)
│   ├── preprocess.py                 # Feature engineering, CKD staging, TII computation
│   ├── eda.py                        # Exploratory data analysis and cohort characterisation
│   ├── baseline_model.py             # Model 1: Random Forest (5-fold CV, SHAP analysis)
│   ├── lstm_model.py                 # Models 2 & 3: BiLSTM and BiLSTM+TII
│   ├── transformer_model.py          # Model 4: Transformer Encoder (TE)
│   └── comparison_and_explainer.py   # 4-way comparison, SHAP, gradient saliency maps
├── backend/
│   └── app.py                        # Flask REST API (serves all 4 models in parallel)
├── frontend/
│   └── dashboard.html                # Clinical decision support dashboard
├── data/                             # Generated datasets (gitignored)
├── models/                           # Trained model artifacts (gitignored)
├── plots/                            # Evaluation plots (ROC, PR, confusion matrix, SHAP)
├── results/                          # Metrics CSVs and clinical explanations
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

3. **Preprocess, engineer features, and compute TII:**
   ```bash
   python src/preprocess.py
   ```

4. **Run exploratory data analysis:**
   ```bash
   python src/eda.py
   ```

5. **Train the RF baseline (with SHAP):**
   ```bash
   python src/baseline_model.py
   ```

6. **Train BiLSTM and BiLSTM+TII models:**
   ```bash
   python src/lstm_model.py
   ```

7. **Train the Transformer Encoder:**
   ```bash
   python src/transformer_model.py
   ```

8. **Run 4-way comparison, SHAP, and gradient saliency:**
   ```bash
   python src/comparison_and_explainer.py
   ```

9. **Launch the clinical dashboard (optional):**
   ```bash
   python backend/app.py
   # Visit http://localhost:5000
   ```

---

## 🧠 Key Technical Details

### TII Computation
```
TII_egfr        = std(eGFR across all visits)        / mean(eGFR across all visits)
TII_creatinine  = std(creatinine across all visits)  / mean(creatinine across all visits)
TII_systolic_bp = std(systolic_bp across all visits) / mean(systolic_bp across all visits)
```
These three scalar values are appended to the feature vector of the **last time step** of each patient's visit sequence before being passed into the BiLSTM model (Equation 8 in the paper).

### BiLSTM Architecture
```
Input → Masking(0.0) → BiLSTM(64, forward+backward) → Dropout(0.3) → Dense(16, ReLU, L2=1e-4) → Sigmoid
```
- Optimizer: Adam (lr = 1×10⁻³)
- Loss: Binary cross-entropy
- Early stopping: patience = 10, monitor = val-AUC-ROC
- Class imbalance handled via inverse-frequency sample weighting

### Transformer Encoder Architecture
```
Input → Dense projection (d=64) → Sinusoidal positional encoding →
2× Multi-Head Attention (4 heads, FFN=256, dropout=0.1) →
Global Average Pooling → Dense classification head
```

### Training Split
- 70% train / 15% validation / 15% test (stratified at patient level)
- RF baseline: 5-fold stratified cross-validation
- All deep models evaluated on the same held-out test set

### Interpretability
- **Global (RF)**: SHAP TreeExplainer on full test cohort — formally consistent feature importance via cooperative game theory axioms
- **Local + Population (BiLSTM+TII)**: Gradient attribution per patient; population-level averaged saliency maps over 100 randomly selected test patients
- **Named risk signals**: TII components (eGFR-CV, creatinine-CV, SBP-CV) are pre-computed and surfaced directly on the dashboard alongside the risk score

---

## 📈 Results Summary

| Model | Accuracy | AUC-ROC | Precision | Recall | F1 |
|-------|----------|---------|-----------|--------|----|
| RF Baseline (Last-Visit) | 0.735 | 0.795 | 0.759 | 0.790 | 0.774 |
| BiLSTM (Time-Series) | 0.979 | 0.998 | 0.984 | 0.979 | 0.982 |
| Transformer Encoder | 0.958 | 0.995 | 0.962 | 0.966 | 0.964 |
| **BiLSTM+TII (Proposed)** | **0.957** | **0.994** | **0.953** | **0.973** | **0.963** |

BiLSTM+TII achieves the highest recall of all models and is the only deep model offering named, auditable clinical risk signals.

---

## 📂 Outputs

| Output | Path |
|--------|------|
| Extracted dataset | `data/ckd_longitudinal.csv` |
| Preprocessed data | `data/ckd_preprocessed.csv` |
| BiLSTM sequences | `data/ckd_sequences.npz` |
| BiLSTM+TII sequences | `data/ckd_sequences_tii.npz` |
| RF baseline model | `models/baseline_rf_model.pkl` |
| BiLSTM model | `models/bilstm_model.h5` |
| BiLSTM+TII model | `models/bilstm_tii_model.h5` |
| Transformer Encoder model | `models/transformer_model.h5` |
| ROC / PR / confusion matrix plots | `plots/comparison_*.png` |
| SHAP beeswarm plot | `plots/shap_beeswarm.png` |
| Gradient saliency maps | `plots/saliency_*.png` |
| Metrics table | `results/model_comparison.csv` |
| Clinical explanations | `results/example_explanations.json` |

---

## 🌐 Flask API

The REST API (`backend/app.py`) accepts a patient's visit sequence as JSON and returns:
- Predicted CKD stage
- Progression probabilities from all **4 models in parallel**
- Plain-language clinical explanation derived from gradient attribution scores
- TII component values (eGFR-CV, creatinine-CV, SBP-CV) for direct clinical inspection

---

## 📚 References

- **MIMIC-IV**: Johnson et al., *MIMIC-IV, a freely accessible electronic health record dataset.* Sci Data, 2023.
- **CKD-EPI 2021**: Inker et al., *New Creatinine- and Cystatin C-Based Equations.* NEJM, 2021.
- **KDIGO 2012**: KDIGO Clinical Practice Guideline for the Evaluation and Management of CKD.
- **SHAP**: Lundberg & Lee, *A unified approach to interpreting model predictions.* NeurIPS, 2017.
- **TII Concept**: Coefficient of Variation as a clinically interpretable measure of longitudinal biomarker instability.