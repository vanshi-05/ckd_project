"""
================================================================================
CKD PROGRESSION PREDICTION PROJECT
Flask REST API + Frontend Server — UPGRADED
Additions:
  - /api/upload_csv  → batch predict from uploaded CSV
================================================================================
"""

import os, sys, io
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

app = Flask(__name__)
CORS(app)

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, 'frontend')

# ─────────────────────────────────────────────────────────────
# SERVE FRONTEND
# ─────────────────────────────────────────────────────────────
@app.route('/')
def serve_dashboard():
    return send_from_directory(FRONTEND_DIR, 'dashboard.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(FRONTEND_DIR, path)

# ─────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────
print("[API] Loading models...")

try:
    RF_MODEL      = joblib.load(os.path.join(BASE_DIR, 'models/baseline_rf_model.pkl'))
    SCALER        = joblib.load(os.path.join(BASE_DIR, 'models/scaler.pkl'))
    FEATURE_NAMES = joblib.load(os.path.join(BASE_DIR, 'models/feature_names.pkl'))
    print("  RF model + scaler loaded")
except Exception as e:
    print(f"  Model load error: {e}")
    RF_MODEL = SCALER = FEATURE_NAMES = None

LSTM_AVAILABLE = TII_AVAILABLE = False
LSTM_MODEL = LSTM_TII_MODEL = None

try:
    import tensorflow as tf
    LSTM_MODEL     = tf.keras.models.load_model(os.path.join(BASE_DIR, 'models/lstm_model.h5'))
    LSTM_AVAILABLE = True
    print("  LSTM loaded")
except Exception as e:
    print(f"  LSTM not loaded (demo mode): {e}")

try:
    import tensorflow as tf
    LSTM_TII_MODEL = tf.keras.models.load_model(os.path.join(BASE_DIR, 'models/lstm_tii_model.h5'))
    TII_AVAILABLE  = True
    print("  LSTM+TII loaded")
except Exception as e:
    print(f"  LSTM+TII not loaded (demo mode): {e}")

try:
    DF = pd.read_csv(os.path.join(BASE_DIR, 'data/ckd_longitudinal.csv'), parse_dates=['visit_date'])
    DF = DF.sort_values(['patient_id', 'visit_date']).reset_index(drop=True)
    DATA_AVAILABLE = True
    print(f"  Dataset: {len(DF):,} visits")
except Exception as e:
    print(f"  Dataset error: {e}")
    DF = None
    DATA_AVAILABLE = False

SEQUENCE_LEN = 5
try:
    if LSTM_AVAILABLE:
        SEQUENCE_LEN = LSTM_MODEL.input_shape[1]
except:
    pass

STAGE_LABELS = {1:"Stage 1 (eGFR >= 90)", 2:"Stage 2 (eGFR 60-89)",
                3:"Stage 3 (eGFR 30-59)", 4:"Stage 4 (eGFR 15-29)", 5:"Stage 5 (eGFR < 15)"}

REQUIRED_CSV_COLS = ['patient_id','age','gender','egfr','creatinine',
                     'systolic_bp','diastolic_bp','hba1c','hemoglobin']

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def egfr_to_stage(egfr):
    if egfr >= 90: return 1
    if egfr >= 60: return 2
    if egfr >= 30: return 3
    if egfr >= 15: return 4
    return 5

def build_feature_vector(visit):
    gender_enc = 1 if str(visit.get('gender','M')).upper() in ('M','MALE','1') else 0
    vec = np.array([[
        float(visit.get('egfr', 60)),
        float(visit.get('creatinine', 1.0)),
        float(visit.get('systolic_bp', 130)),
        float(visit.get('diastolic_bp', 85)),
        float(visit.get('hba1c', 5.5)),
        float(visit.get('hemoglobin', 13.0)),
        float(visit.get('age', 55)),
        gender_enc
    ]], dtype=np.float32)
    if SCALER is not None:
        vec = SCALER.transform(vec)
    return vec

def build_lstm_sequence(visits):
    seq    = np.zeros((SEQUENCE_LEN, 8), dtype=np.float32)
    recent = visits[-SEQUENCE_LEN:]
    offset = SEQUENCE_LEN - len(recent)
    for i, v in enumerate(recent):
        seq[offset+i] = build_feature_vector(v)[0]
    return seq[np.newaxis, :, :]

def compute_tii(values):
    arr  = np.array(values, dtype=float)
    mean = np.mean(np.abs(arr))
    return float(np.std(arr) / max(mean, 1e-6))

def get_risk_label(score):
    if score >= 0.65: return 'HIGH'
    if score >= 0.35: return 'MEDIUM'
    return 'LOW'

def predict_for_visits(visits):
    if not visits:
        return None
    last = visits[-1]
    vec  = build_feature_vector(last)

    baseline_risk = float(RF_MODEL.predict_proba(vec)[0][1]) if RF_MODEL else 0.5

    if LSTM_AVAILABLE:
        seq       = build_lstm_sequence(visits)
        lstm_risk = float(LSTM_MODEL.predict(seq, verbose=0)[0][0])
    else:
        egfrs = [float(v.get('egfr', 60)) for v in visits[-5:]]
        slope = np.polyfit(range(len(egfrs)), egfrs, 1)[0] if len(egfrs) >= 2 else 0
        lstm_risk = float(np.clip(0.5 - slope * 0.08, 0.05, 0.95))

    if TII_AVAILABLE:
        try:
            seq  = build_lstm_sequence(visits)
            egfrs = [float(v.get('egfr', 60)) for v in visits]
            crs   = [float(v.get('creatinine', 1.0)) for v in visits]
            sbps  = [float(v.get('systolic_bp', 130)) for v in visits]
            tii_feats = np.full((1, SEQUENCE_LEN, 3),
                                [compute_tii(egfrs), compute_tii(crs), compute_tii(sbps)],
                                dtype=np.float32)
            seq_tii  = np.concatenate([seq, tii_feats], axis=2)
            tii_risk = float(LSTM_TII_MODEL.predict(seq_tii, verbose=0)[0][0])
        except:
            tii_risk = lstm_risk
    else:
        egfrs    = [float(v.get('egfr', 60)) for v in visits]
        tii_bump = compute_tii(egfrs) * 0.15
        tii_risk = float(np.clip(lstm_risk + tii_bump, 0.0, 0.99))

    stage = egfr_to_stage(float(last.get('egfr', 60)))
    return {
        "baseline_risk": round(baseline_risk, 4),
        "lstm_risk":     round(lstm_risk, 4),
        "tii_risk":      round(tii_risk, 4),
        "risk_label":    get_risk_label(tii_risk),
        "ckd_stage":     stage,
        "stage_label":   STAGE_LABELS.get(stage, "Unknown"),
        "models_used":   {"baseline": RF_MODEL is not None,
                          "lstm": LSTM_AVAILABLE, "lstm_tii": TII_AVAILABLE}
    }

# ─────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────
@app.route('/health')
def health():
    return jsonify({"status":"ok","rf":RF_MODEL is not None,
                    "lstm":LSTM_AVAILABLE,"tii":TII_AVAILABLE,"data":DATA_AVAILABLE})

@app.route('/api/patients')
def get_patients():
    if not DATA_AVAILABLE:
        return jsonify({"error":"Dataset not loaded"}), 503
    return jsonify(DF['patient_id'].unique().tolist())

@app.route('/api/patient/<patient_id>')
def get_patient(patient_id):
    if not DATA_AVAILABLE:
        return jsonify({"error":"Dataset not loaded"}), 503
    df = DF[DF['patient_id'] == patient_id]
    if df.empty:
        return jsonify({"error":"Patient not found"}), 404
    df = df.fillna(0)
    df['visit_date'] = df['visit_date'].astype(str)
    return jsonify(df.to_dict(orient='records'))

@app.route('/api/predict', methods=['POST'])
def predict():
    if RF_MODEL is None:
        return jsonify({"error":"Model not loaded"}), 503
    data   = request.get_json()
    visits = data.get("visits", [])
    if not visits:
        return jsonify({"error":"visits required"}), 400
    return jsonify(predict_for_visits(visits))

# ─────────────────────────────────────────────────────────────
# CSV UPLOAD ENDPOINT
# ─────────────────────────────────────────────────────────────
@app.route('/api/upload_csv', methods=['POST'])
def upload_csv():
    """
    Hospital uploads a CSV. Returns predictions for all patients.

    Required CSV columns:
      patient_id, age, gender, egfr, creatinine,
      systolic_bp, diastolic_bp, hba1c, hemoglobin

    Optional: visit_date, visit_number
    """
    if RF_MODEL is None:
        return jsonify({"error": "Model not loaded. Run the training pipeline first."}), 503

    if 'file' not in request.files:
        return jsonify({"error": "No file. Send multipart/form-data with key 'file'."}), 400

    file = request.files['file']
    if not file.filename.lower().endswith('.csv'):
        return jsonify({"error": "Only .csv files accepted."}), 400

    try:
        content = file.read().decode('utf-8')
        df = pd.read_csv(io.StringIO(content))
    except Exception as e:
        return jsonify({"error": f"CSV parse error: {str(e)}"}), 400

    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

    missing = [c for c in REQUIRED_CSV_COLS if c not in df.columns]
    if missing:
        return jsonify({
            "error": f"Missing columns: {missing}",
            "required": REQUIRED_CSV_COLS,
            "found": list(df.columns)
        }), 400

    if 'visit_date' not in df.columns:
        df['visit_date'] = '2024-01-01'
    if 'visit_number' not in df.columns:
        df['visit_number'] = df.groupby('patient_id').cumcount() + 1

    df = df.fillna({'egfr':60,'creatinine':1.0,'systolic_bp':130,
                    'diastolic_bp':85,'hba1c':5.5,'hemoglobin':13.0,
                    'age':55,'gender':'M'})

    results = []
    for pid, group in df.groupby('patient_id'):
        group  = group.sort_values('visit_number')
        visits = group.to_dict(orient='records')
        pred   = predict_for_visits(visits)
        last   = visits[-1]

        egfrs = [float(v.get('egfr', 60)) for v in visits]
        slope = np.polyfit(range(len(egfrs)), egfrs, 1)[0] if len(egfrs) >= 2 else 0
        tii_e = compute_tii(egfrs)
        explanation = (
            f"eGFR trend: {slope:+.2f} ml/min per visit over {len(visits)} visits "
            f"(latest eGFR: {float(last.get('egfr',60)):.1f}). "
            f"Biomarker instability (TII_eGFR={tii_e:.3f}). "
            f"LSTM+TII risk: {pred['tii_risk']*100:.1f}%."
        )

        results.append({
            "patient_id":    str(pid),
            "age":           float(last.get('age', 0)),
            "gender":        str(last.get('gender', 'U')),
            "visit_count":   len(visits),
            "latest_egfr":   float(last.get('egfr', 0)),
            "ckd_stage":     pred['ckd_stage'],
            "stage_label":   pred['stage_label'],
            "baseline_risk": pred['baseline_risk'],
            "lstm_risk":     pred['lstm_risk'],
            "tii_risk":      pred['tii_risk'],
            "risk_label":    pred['risk_label'],
            "explanation":   explanation,
            "visits": [{
                "n":     int(v.get('visit_number', i+1)),
                "date":  str(v.get('visit_date', '')),
                "egfr":  float(v.get('egfr', 0)),
                "cr":    float(v.get('creatinine', 0)),
                "sbp":   float(v.get('systolic_bp', 0)),
                "hba1c": float(v.get('hba1c', 0)),
                "hgb":   float(v.get('hemoglobin', 0)),
                "stage": egfr_to_stage(float(v.get('egfr', 60))),
                "prog":  0
            } for i, v in enumerate(visits)]
        })

    results.sort(key=lambda x: x['tii_risk'], reverse=True)

    return jsonify({
        "total_patients": len(results),
        "high_risk":   sum(1 for r in results if r['risk_label'] == 'HIGH'),
        "medium_risk": sum(1 for r in results if r['risk_label'] == 'MEDIUM'),
        "low_risk":    sum(1 for r in results if r['risk_label'] == 'LOW'),
        "patients":    results
    })

# ─────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n===========================================")
    print(" CKD Prediction API — http://localhost:5000")
    print("===========================================\n")
    app.run(debug=False, host='0.0.0.0', port=5000)