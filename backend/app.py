"""
================================================================================
CKD PROGRESSION PREDICTION PROJECT
Flask REST API + Frontend Server
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ─────────────────────────────────────────────────────────────
# INITIAL SETUP
# ─────────────────────────────────────────────────────────────

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, 'frontend')

# ─────────────────────────────────────────────────────────────
# SERVE FRONTEND
# ─────────────────────────────────────────────────────────────

@app.route('/')
def serve_dashboard():
    return send_from_directory(FRONTEND_DIR, 'dashboard.html')

@app.route('/<path:path>')
def serve_static_files(path):
    return send_from_directory(FRONTEND_DIR, path)

# ─────────────────────────────────────────────────────────────
# LOAD MODELS & DATA
# ─────────────────────────────────────────────────────────────

print("[API] Loading models and data...")

try:
    RF_MODEL      = joblib.load(os.path.join(BASE_DIR, 'models/baseline_rf_model.pkl'))
    SCALER        = joblib.load(os.path.join(BASE_DIR, 'models/scaler.pkl'))
    FEATURE_NAMES = joblib.load(os.path.join(BASE_DIR, 'models/feature_names.pkl'))
    print("  ✅ Random Forest model loaded")
    print("  ✅ Scaler loaded")
except Exception as e:
    print(f"  ❌ Model load error: {e}")
    RF_MODEL = SCALER = FEATURE_NAMES = None

try:
    LSTM_MODEL = tf.keras.models.load_model(
        os.path.join(BASE_DIR, 'models/lstm_model.h5')
    )
    print("  ✅ LSTM model loaded")
    LSTM_AVAILABLE = True
except Exception as e:
    LSTM_MODEL     = None
    LSTM_AVAILABLE = False
    print(f"  ❌ LSTM load error: {e}")
    print("  ⚠️ LSTM not loaded (Demo mode active)")

try:
    LSTM_TII_MODEL = tf.keras.models.load_model(
        os.path.join(BASE_DIR, 'models/lstm_tii_model.h5')
    )
    print("  ✅ LSTM+TII model loaded")
    TII_AVAILABLE = True
except Exception as e:
    LSTM_TII_MODEL = None
    TII_AVAILABLE  = False
    print(f"  ❌ LSTM+TII load error: {e}")
    print("  ⚠️ LSTM+TII not loaded (Demo mode active)")

try:
    DF = pd.read_csv(
        os.path.join(BASE_DIR, 'data/ckd_longitudinal.csv'),
        parse_dates=['visit_date']
    )
    DF = DF.sort_values(['patient_id', 'visit_date']).reset_index(drop=True)
    DATA_AVAILABLE = True
    print(f"  ✅ Dataset loaded: {len(DF):,} visits")
except Exception as e:
    print(f"  ❌ Dataset load error: {e}")
    DF = None
    DATA_AVAILABLE = False

STAGE_LABELS = {
    1: "Stage 1 (eGFR ≥ 90)",
    2: "Stage 2 (eGFR 60–89)",
    3: "Stage 3 (eGFR 30–59)",
    4: "Stage 4 (eGFR 15–29)",
    5: "Stage 5 (eGFR < 15)"
}

# Dynamic: detect from saved model input shape, with fallback
try:
    SEQUENCE_LEN = LSTM_MODEL.input_shape[1] if LSTM_AVAILABLE else 5
except Exception:
    SEQUENCE_LEN = 5
print(f"  Sequence length: {SEQUENCE_LEN}")

# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────

def egfr_to_stage(egfr):
    if egfr >= 90: return 1
    if egfr >= 60: return 2
    if egfr >= 30: return 3
    if egfr >= 15: return 4
    return 5


def build_feature_vector(visit):
    gender_enc = 1 if str(visit.get('gender', 'M')).upper() == 'M' else 0
    vec = np.array([[ 
        visit.get('egfr', 60),
        visit.get('creatinine', 1.0),
        visit.get('systolic_bp', 130),
        visit.get('diastolic_bp', 85),
        visit.get('hba1c', 5.5),
        visit.get('hemoglobin', 13.0),
        visit.get('age', 55),
        gender_enc
    ]], dtype=np.float32)

    if SCALER is not None:
        vec = SCALER.transform(vec)

    return vec


def build_lstm_sequence(visits):
    seq = np.zeros((SEQUENCE_LEN, 8), dtype=np.float32)
    recent = visits[-SEQUENCE_LEN:]
    offset = SEQUENCE_LEN - len(recent)

    for i, visit in enumerate(recent):
        seq[offset+i] = build_feature_vector(visit)[0]

    return seq[np.newaxis, :, :]


def get_lstm_risk(visits):
    if LSTM_AVAILABLE:
        seq = build_lstm_sequence(visits)
        risk = float(LSTM_MODEL.predict(seq, verbose=0)[0][0])
    else:
        egfr_vals = [v.get('egfr', 60) for v in visits[-5:]]
        if len(egfr_vals) >= 2:
            slope = np.polyfit(range(len(egfr_vals)), egfr_vals, 1)[0]
            risk = np.clip(0.5 - slope * 0.08, 0.05, 0.95)
        else:
            risk = 0.3
    return round(float(risk), 4)

# ─────────────────────────────────────────────────────────────
# API ENDPOINTS
# ─────────────────────────────────────────────────────────────

@app.route('/health')
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": RF_MODEL is not None,
        "lstm_available": LSTM_AVAILABLE,
        "tii_available": TII_AVAILABLE,
        "data_loaded": DATA_AVAILABLE
    })


@app.route('/api/patients')
def get_patients():
    if not DATA_AVAILABLE:
        return jsonify({"error": "Dataset not loaded"}), 503

    patients = DF['patient_id'].unique().tolist()
    return jsonify(patients)


@app.route('/api/patient/<patient_id>')
def get_patient(patient_id):
    if not DATA_AVAILABLE:
        return jsonify({"error": "Dataset not loaded"}), 503

    df = DF[DF['patient_id'] == patient_id]
    if df.empty:
        return jsonify({"error": "Patient not found"}), 404

    df = df.fillna("null")
    df['visit_date'] = df['visit_date'].astype(str)

    return jsonify(df.to_dict(orient='records'))


@app.route('/api/predict', methods=['POST'])
def predict():
    if RF_MODEL is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json()
    visits = data.get("visits", [])

    if not visits:
        return jsonify({"error": "Visits required"}), 400

    last_visit = visits[-1]
    vec = build_feature_vector(last_visit)

    baseline_risk = float(RF_MODEL.predict_proba(vec)[0][1])
    lstm_risk = get_lstm_risk(visits)
    tii_risk  = lstm_risk  # Fallback: same as LSTM

    # If TII model is available, use it
    if TII_AVAILABLE and LSTM_TII_MODEL is not None:
        try:
            seq = build_lstm_sequence(visits)
            # TII model has 11 features instead of 8 — pad with TII values
            # For live prediction, TII is computed from the visit history
            egfr_vals = [v.get('egfr', 60) for v in visits]
            cr_vals = [v.get('creatinine', 1.0) for v in visits]
            sbp_vals = [v.get('systolic_bp', 130) for v in visits]
            
            tii_egfr = np.std(egfr_vals) / max(np.mean(np.abs(egfr_vals)), 1e-6)
            tii_cr   = np.std(cr_vals) / max(np.mean(np.abs(cr_vals)), 1e-6)
            tii_sbp  = np.std(sbp_vals) / max(np.mean(np.abs(sbp_vals)), 1e-6)
            
            tii_features = np.full((1, SEQUENCE_LEN, 3), [tii_egfr, tii_cr, tii_sbp], dtype=np.float32)
            seq_tii = np.concatenate([seq, tii_features], axis=2)
            tii_risk = float(LSTM_TII_MODEL.predict(seq_tii, verbose=0)[0][0])
        except Exception:
            tii_risk = lstm_risk

    stage = egfr_to_stage(last_visit.get("egfr", 60))

    return jsonify({
        "baseline_risk": round(baseline_risk, 4),
        "lstm_risk": round(lstm_risk, 4),
        "tii_risk": round(tii_risk, 4),
        "ckd_stage": stage,
        "stage_label": STAGE_LABELS.get(stage),
        "models_used": {
            "baseline": True,
            "lstm": LSTM_AVAILABLE,
            "lstm_tii": TII_AVAILABLE
        }
    })


# ─────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n===========================================")
    print(" CKD Prediction API running")
    print(" http://localhost:5000")
    print("===========================================\n")

    app.run(debug=False, host="0.0.0.0", port=5000)

