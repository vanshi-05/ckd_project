"""
================================================================================
CKD PROGRESSION PREDICTION PROJECT
Phase 2: Data Preprocessing & Time-Series Sequence Construction

Author      : B.Tech CSE (AIML) Minor Project
Description : Full preprocessing pipeline including:
              1. Missing value handling (forward-fill within patient)
              2. Feature engineering
              3. Train/test split at PATIENT level (no leakage)
              4. Min-Max normalization (fit on train only)
              5. LSTM-ready padded sequence construction
              6. Baseline flat feature vector construction

Outputs:
  data/ckd_preprocessed.csv      → Cleaned flat data
  data/ckd_sequences.npz         → LSTM sequences (X_train, y_train, X_test, y_test)
  data/ckd_baseline.npz          → Baseline flat features (last-visit only)
  models/scaler.pkl              → Fitted MinMaxScaler (for API inference)
  models/feature_names.pkl       → Feature column names
================================================================================
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

# Features used for modeling (ordered consistently)
FEATURE_COLS = [
    'egfr',          # Primary CKD biomarker
    'creatinine',    # Kidney function proxy
    'systolic_bp',   # Blood pressure
    'diastolic_bp',  # Blood pressure
    'hba1c',         # Diabetes control
    'hemoglobin',    # Anemia marker
    'age',           # Demographic risk factor
    'gender_encoded' # 0=Female, 1=Male
]

# Enhanced feature set WITH Temporal Instability Index (TII)
# TII = std(biomarker) / mean(biomarker) per patient (Coefficient of Variation)
# This is the KEY NOVELTY of the project — captures biomarker volatility
FEATURE_COLS_TII = FEATURE_COLS + [
    'tii_egfr',        # eGFR instability
    'tii_creatinine',  # Creatinine instability
    'tii_systolic_bp', # Blood pressure instability
]

TARGET_COL   = 'progression_label'
SEQUENCE_LEN = 5      # Default, overridden dynamically at runtime
MISSING_THRESHOLD = 0.40  # Exclude patients with >40% missing values
RANDOM_STATE = 42


# ─────────────────────────────────────────────────────────────
# STEP 1: Load raw data
# ─────────────────────────────────────────────────────────────
def load_data(path: str = 'data/ckd_longitudinal.csv') -> pd.DataFrame:
    """Load raw longitudinal CKD dataset."""
    print("\n[STEP 1] Loading raw dataset...")
    df = pd.read_csv(path, parse_dates=['visit_date'])
    df = df.sort_values(['patient_id', 'visit_date']).reset_index(drop=True)
    print(f"  Shape       : {df.shape}")
    print(f"  Patients    : {df['patient_id'].nunique()}")
    print(f"  Date range  : {df['visit_date'].min().date()} → {df['visit_date'].max().date()}")
    print(f"  Missing vals: {df.isnull().sum().sum()}")
    return df


# ─────────────────────────────────────────────────────────────
# STEP 2: Remove patients with excessive missing data
# ─────────────────────────────────────────────────────────────
def filter_patients(df: pd.DataFrame) -> pd.DataFrame:
    """
    Exclude patients where > MISSING_THRESHOLD fraction of
    key lab values are missing (unreliable longitudinal record).
    """
    print(f"\n[STEP 2] Filtering patients with >{MISSING_THRESHOLD:.0%} missing values...")
    initial_patients = df['patient_id'].nunique()

    # Per-patient missing rate on key clinical columns
    key_cols = ['egfr', 'creatinine', 'systolic_bp']
    patient_missing = df.groupby('patient_id')[key_cols].apply(
        lambda x: x.isnull().mean().mean()
    )
    valid_patients = patient_missing[patient_missing <= MISSING_THRESHOLD].index
    df_filtered = df[df['patient_id'].isin(valid_patients)].copy()

    removed = initial_patients - len(valid_patients)
    print(f"  Removed {removed} patients | Kept {len(valid_patients)} patients")
    return df_filtered


# ─────────────────────────────────────────────────────────────
# STEP 3: Encode categorical variables
# ─────────────────────────────────────────────────────────────
def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode gender: M=1, F=0. Binary label, no one-hot needed."""
    print("\n[STEP 3] Encoding categorical features...")
    df = df.copy()
    df['gender_encoded'] = (df['gender'] == 'M').astype(int)
    print(f"  Gender encoded: M=1, F=0")
    print(f"  Male patients  : {(df['gender'] == 'M').sum():,} visits")
    print(f"  Female patients: {(df['gender'] == 'F').sum():,} visits")
    return df


# ─────────────────────────────────────────────────────────────
# STEP 4: Forward-fill missing values within each patient
# ─────────────────────────────────────────────────────────────
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strategy: Forward-fill (LOCF - Last Observation Carried Forward)
    within each patient's time series.

    Clinical justification: Lab values change gradually in CKD.
    If a value is missing at visit 4, using visit 3's value is a
    much better estimate than using a global population mean.

    Remaining NaN after forward-fill (i.e., missing at FIRST visit):
    → Backfill using the patient's next available value.
    → If still NaN: use column median (last resort, rare cases).
    """
    print("\n[STEP 4] Handling missing values (LOCF strategy)...")
    missing_before = df[FEATURE_COLS].isnull().sum().sum()
    df = df.copy()

    # Apply within-patient forward-fill then backfill
    fill_cols = ['egfr', 'creatinine', 'systolic_bp', 'diastolic_bp',
                 'hba1c', 'hemoglobin']

    df[fill_cols] = (
        df.groupby('patient_id')[fill_cols]
          .transform(lambda x: x.ffill().bfill())
    )

    # Final fallback: global column median for any remaining NaN
    col_medians = df[fill_cols].median()
    df[fill_cols] = df[fill_cols].fillna(col_medians)

    missing_after = df[FEATURE_COLS].isnull().sum().sum()
    print(f"  Missing before: {missing_before:,}")
    print(f"  Missing after : {missing_after:,}")
    print(f"  Strategy      : LOCF → Backfill → Column Median")
    return df


# ─────────────────────────────────────────────────────────────
# STEP 5: Patient-level train/test split (NO DATA LEAKAGE)
# ─────────────────────────────────────────────────────────────
def split_patients(df: pd.DataFrame) -> tuple:
    """
    CRITICAL: Split at the PATIENT level, NOT the visit level.

    Why: If we split by visit, the same patient could have visits
    in both train and test sets. The model would essentially
    memorize patient-specific patterns → inflated test performance.

    Strategy: 80% patients → train, 20% patients → test.
    Stratified by initial CKD stage to maintain stage distribution.
    """
    print("\n[STEP 5] Patient-level train/test split (80/20, stratified)...")

    # Get one representative row per patient for stratification
    patient_profiles = (
        df.groupby('patient_id')
          .agg(initial_stage=('ckd_stage', 'first'))
          .reset_index()
    )

    train_patients, test_patients = train_test_split(
        patient_profiles['patient_id'],
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=patient_profiles['initial_stage']
    )

    df_train = df[df['patient_id'].isin(train_patients)].copy()
    df_test  = df[df['patient_id'].isin(test_patients)].copy()

    print(f"  Train patients : {df_train['patient_id'].nunique():3d}  "
          f"({df_train['patient_id'].nunique()/df['patient_id'].nunique():.0%})")
    print(f"  Test  patients : {df_test['patient_id'].nunique():3d}  "
          f"({df_test['patient_id'].nunique()/df['patient_id'].nunique():.0%})")
    print(f"  Train visits   : {len(df_train):,}")
    print(f"  Test  visits   : {len(df_test):,}")
    print(f"  Train progression rate: {df_train[TARGET_COL].mean():.1%}")
    print(f"  Test  progression rate: {df_test[TARGET_COL].mean():.1%}")

    return df_train, df_test


# ─────────────────────────────────────────────────────────────
# STEP 6: Fit scaler on TRAIN data, transform both sets
# ─────────────────────────────────────────────────────────────
def normalize_features(df_train: pd.DataFrame,
                        df_test: pd.DataFrame,
                        scaler_path: str = 'models/scaler.pkl') -> tuple:
    """
    Min-Max normalization to [0, 1] range.

    CRITICAL: Scaler is FIT ONLY on training data.
    Test data is transformed using train-fitted scaler.
    This prevents data leakage of test statistics into model training.

    The scaler is saved for use during API inference.
    """
    print("\n[STEP 6] Min-Max Normalization (fit on train only)...")
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)

    scaler = MinMaxScaler(feature_range=(0, 1))

    # Fit ONLY on training feature values
    scaler.fit(df_train[FEATURE_COLS])

    # Transform both sets
    df_train = df_train.copy()
    df_test  = df_test.copy()
    df_train[FEATURE_COLS] = scaler.transform(df_train[FEATURE_COLS])
    df_test[FEATURE_COLS]  = scaler.transform(df_test[FEATURE_COLS])

    # Save scaler for inference
    joblib.dump(scaler, scaler_path)
    joblib.dump(FEATURE_COLS, 'models/feature_names.pkl')

    print(f"  Features scaled : {len(FEATURE_COLS)}")
    print(f"  Scaler saved    → {scaler_path}")

    # Print original vs scaled range for each feature
    for i, col in enumerate(FEATURE_COLS):
        print(f"    {col:20s}: [{scaler.data_min_[i]:.2f}, {scaler.data_max_[i]:.2f}]"
              f" → [0.0, 1.0]")

    return df_train, df_test, scaler


# ─────────────────────────────────────────────────────────────
# STEP 7: Construct LSTM sequences with DYNAMIC TIME PADDING
# ─────────────────────────────────────────────────────────────
def build_lstm_sequences(df: pd.DataFrame) -> tuple:
    """
    Build sequence tensors for LSTM training using dynamic padding.
    
    Instead of hardcoding a 5-visit window and truncating data, we now:
    1. Find the maximum number of visits for any patient in the dataset.
    2. Zero-pad all sequences to match this maximum sequence length.
    3. The LSTM's Masking layer will efficiently ignore the zero-padded steps.

    Returns:
    --------
    X : np.ndarray of shape (n_patients, max_seq_len, n_features)
    y : np.ndarray of shape (n_patients,) — binary progression labels at their last visit
    seq_len: int — The dynamic sequence length determined
    """
    # 1. Determine dynamic sequence length (the longest history in the building)
    max_seq_len = df.groupby('patient_id').size().max()
    print(f"  → Dynamic max_sequence_length detected: {max_seq_len}")
    
    X_list, y_list = [], []

    for patient_id, group in df.groupby('patient_id'):
        group = group.sort_values('visit_date').reset_index(drop=True)
        features = group[FEATURE_COLS].values   # (n_visits, n_features)
        labels   = group[TARGET_COL].values     # (n_visits,)
        
        # Target is whether they progressed during their last observed visit transition
        # Alternatively, did they progress AT ALL across their history? We use the max label.
        patient_target = 1 if (labels == 1).any() else 0
        
        # Left-pad sequence to max_seq_len
        n_visits = len(features)
        pad_len = max_seq_len - n_visits
        if pad_len > 0:
            padding = np.zeros((pad_len, len(FEATURE_COLS)))
            padded_features = np.vstack([padding, features])
        else:
            padded_features = features
            
        X_list.append(padded_features)
        y_list.append(patient_target)

    X = np.array(X_list, dtype=np.float32)   # (n_patients, max_seq_len, n_features)
    y = np.array(y_list, dtype=np.float32)   # (n_patients,)

    return X, y, max_seq_len


# ─────────────────────────────────────────────────────────────
# STEP 8: Construct Baseline (last-visit only) flat features
# ─────────────────────────────────────────────────────────────
def build_baseline_features(df: pd.DataFrame) -> tuple:
    """
    Baseline model input: ONLY the most recent visit's features.

    This deliberately ignores all temporal history — it mirrors
    how most clinical risk tools work today (single-point-in-time).

    Returns:
    --------
    X : np.ndarray of shape (n_patients, n_features)
    y : np.ndarray of shape (n_patients,) — label at last visit
    """
    # For each patient, keep only the most recent visit's features
    last_visits = (
        df.sort_values('visit_date')
          .groupby('patient_id')
          .last()
          .reset_index()
    )
    X = last_visits[FEATURE_COLS].values.astype(np.float32)
    
    # Target: did this patient EVER progress? (consistent with LSTM target)
    patient_targets = df.groupby('patient_id')[TARGET_COL].max().reset_index()
    patient_targets = patient_targets.set_index('patient_id')
    y = last_visits['patient_id'].map(patient_targets[TARGET_COL]).values.astype(np.float32)

    return X, y


# ─────────────────────────────────────────────────────────────
# STEP 9: Add derived clinical features
# ─────────────────────────────────────────────────────────────
def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add clinically meaningful derived features:
    - eGFR change from last visit (trend signal)
    - BP pulse pressure (sbp - dbp)
    - Temporal Instability Index (TII) for key biomarkers
    """
    df = df.copy()

    # eGFR change from previous visit (within patient)
    df['egfr_change'] = df.groupby('patient_id')['egfr'].diff().fillna(0)

    # Pulse pressure (cardiovascular risk in CKD)
    df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']

    # Days since first visit (temporal feature)
    df['visit_date_parsed'] = pd.to_datetime(df['visit_date'])
    df['days_since_first']  = df.groupby('patient_id')['visit_date_parsed'].transform(
        lambda x: (x - x.min()).dt.days
    )
    df = df.drop(columns=['visit_date_parsed'])

    return df


# ─────────────────────────────────────────────────────────────
# STEP 9b: Compute Temporal Instability Index (TII)
# ─────────────────────────────────────────────────────────────
def compute_tii(df: pd.DataFrame) -> pd.DataFrame:
    """
    Temporal Instability Index (TII) — PROJECT KEY NOVELTY.

    TII = Standard Deviation / Mean  (Coefficient of Variation)
    Computed per patient across ALL their visits for key biomarkers.

    Clinical relevance:
      - Stable biomarker values → Low TII  → lower risk
      - Large fluctuations     → High TII → higher risk of progression

    The TII value is broadcast to every row of a patient so it can be
    used as an additional LSTM input feature at every timestep.
    """
    print("\n[STEP 9b] Computing Temporal Instability Index (TII)...")
    df = df.copy()

    tii_biomarkers = {
        'tii_egfr':        'egfr',
        'tii_creatinine':  'creatinine',
        'tii_systolic_bp': 'systolic_bp',
    }

    for tii_col, raw_col in tii_biomarkers.items():
        patient_stats = df.groupby('patient_id')[raw_col].agg(['std', 'mean'])
        # Avoid division by zero: if mean is 0 or NaN, TII = 0
        patient_stats['tii'] = np.where(
            (patient_stats['mean'].abs() > 1e-6) & patient_stats['std'].notna(),
            patient_stats['std'] / patient_stats['mean'].abs(),
            0.0
        )
        tii_map = patient_stats['tii'].to_dict()
        df[tii_col] = df['patient_id'].map(tii_map).fillna(0.0)
        print(f"  {tii_col:25s}: mean={df[tii_col].mean():.4f}, "
              f"median={df[tii_col].median():.4f}, max={df[tii_col].max():.4f}")

    return df


# ─────────────────────────────────────────────────────────────
# MAIN PREPROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────
def run_preprocessing_pipeline(
        raw_data_path  : str = 'data/ckd_longitudinal.csv',
        output_clean   : str = 'data/ckd_preprocessed.csv',
        output_sequences: str = 'data/ckd_sequences.npz',
        output_baseline: str = 'data/ckd_baseline.npz',
        scaler_path    : str = 'models/scaler.pkl'
) -> dict:
    """
    Run the full preprocessing pipeline.
    Returns a dictionary with all processed data splits.
    """
    print("=" * 65)
    print("  CKD Data Preprocessing Pipeline")
    print("=" * 65)

    # Pipeline execution
    df        = load_data(raw_data_path)
    df        = filter_patients(df)
    df        = encode_features(df)
    df        = handle_missing_values(df)
    df        = add_derived_features(df)
    df        = compute_tii(df)           # ← KEY NOVELTY: Temporal Instability Index
    df_train, df_test = split_patients(df)
    df_train, df_test, scaler = normalize_features(df_train, df_test, scaler_path)

    # Build LSTM sequences
    print("\n[STEP 7] Building LSTM sequences (Dynamic padding to max patient history)...")
    
    # We must ensure both train and test sets are padded to the GLOBALLY maximum sequence length
    global_max_len = max(df_train.groupby('patient_id').size().max(), 
                         df_test.groupby('patient_id').size().max())
                         
    def pad_to_length(df_subset, max_len, feature_cols):
        """Generic padder: works for both standard and TII-enhanced features."""
        X_list, y_list = [], []
        for patient_id, group in df_subset.groupby('patient_id'):
            group = group.sort_values('visit_date').reset_index(drop=True)
            features = group[feature_cols].values
            labels = group[TARGET_COL].values
            patient_target = 1 if (labels == 1).any() else 0
            
            n_visits = len(features)
            pad_len = max_len - n_visits
            if pad_len > 0:
                padding = np.zeros((pad_len, len(feature_cols)))
                padded_features = np.vstack([padding, features])
            else:
                padded_features = features
            X_list.append(padded_features)
            y_list.append(patient_target)
        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

    # ── Model 2: Standard LSTM sequences (without TII) ──
    X_train_seq, y_train_seq = pad_to_length(df_train, global_max_len, FEATURE_COLS)
    X_test_seq,  y_test_seq  = pad_to_length(df_test,  global_max_len, FEATURE_COLS)
    
    SEQUENCE_LEN = global_max_len
    
    print(f"  [LSTM] Train: {X_train_seq.shape}  |  Test: {X_test_seq.shape}")
    print(f"  [LSTM] Train progression: {y_train_seq.mean():.1%}")
    print(f"  [LSTM] Sequence shape: (samples, T={SEQUENCE_LEN}, features={len(FEATURE_COLS)})")

    # ── Model 3: LSTM + TII sequences (PROPOSED METHOD) ──
    print("\n[STEP 7b] Building LSTM+TII sequences (Proposed Method)...")
    X_train_tii, y_train_tii = pad_to_length(df_train, global_max_len, FEATURE_COLS_TII)
    X_test_tii,  y_test_tii  = pad_to_length(df_test,  global_max_len, FEATURE_COLS_TII)
    
    print(f"  [LSTM+TII] Train: {X_train_tii.shape}  |  Test: {X_test_tii.shape}")
    print(f"  [LSTM+TII] Sequence shape: (samples, T={SEQUENCE_LEN}, features={len(FEATURE_COLS_TII)})")

    # Build baseline flat features
    print("\n[STEP 8] Building baseline flat feature vectors (last-visit only)...")
    X_train_base, y_train_base = build_baseline_features(df_train)
    X_test_base,  y_test_base  = build_baseline_features(df_test)
    print(f"  Train baseline: {X_train_base.shape}  →  labels: {y_train_base.shape}")
    print(f"  Test  baseline: {X_test_base.shape}   →  labels: {y_test_base.shape}")

    # Save all artifacts
    print("\n[STEP 10] Saving preprocessed data artifacts...")
    os.makedirs('data', exist_ok=True)

    # Full clean DataFrame
    full_clean = pd.concat([df_train, df_test]).sort_values(['patient_id', 'visit_date'])
    full_clean.to_csv(output_clean, index=False)
    print(f"  Saved: {output_clean}")

    # Standard LSTM sequences
    np.savez_compressed(output_sequences,
                        X_train=X_train_seq, y_train=y_train_seq,
                        X_test=X_test_seq,   y_test=y_test_seq)
    print(f"  Saved: {output_sequences}")

    # TII-enhanced LSTM sequences
    output_tii = output_sequences.replace('.npz', '_tii.npz')
    np.savez_compressed(output_tii,
                        X_train=X_train_tii, y_train=y_train_tii,
                        X_test=X_test_tii,   y_test=y_test_tii)
    print(f"  Saved: {output_tii}")

    # Baseline arrays
    np.savez_compressed(output_baseline,
                        X_train=X_train_base, y_train=y_train_base,
                        X_test=X_test_base,   y_test=y_test_base)
    print(f"  Saved: {output_baseline}")

    # Class imbalance summary (important for model training)
    print("\n[CLASS IMBALANCE ANALYSIS]")
    n_pos = int(y_train_seq.sum())
    n_neg = int(len(y_train_seq) - n_pos)
    ratio = n_neg / max(n_pos, 1)
    print(f"  Train: {n_pos} progression, {n_neg} no-progression")
    print(f"  Imbalance ratio: 1:{ratio:.1f}")
    print(f"  → Use class_weight='balanced' in sklearn models")
    print(f"  → Use class_weight={{0:{1/ratio:.3f}, 1:1.0}} in Keras LSTM")

    print("\n" + "=" * 65)
    print("  ✅ Preprocessing Pipeline Complete")
    print(f"  3 model input sets generated:")
    print(f"    1. Baseline (last-visit):  {output_baseline}")
    print(f"    2. LSTM (standard):        {output_sequences}")
    print(f"    3. LSTM+TII (proposed):    {output_tii}")
    print("=" * 65)

    return {
        'df_train'       : df_train,
        'df_test'        : df_test,
        'X_train_seq'    : X_train_seq,
        'y_train_seq'    : y_train_seq,
        'X_test_seq'     : X_test_seq,
        'y_test_seq'     : y_test_seq,
        'X_train_tii'    : X_train_tii,
        'y_train_tii'    : y_train_tii,
        'X_test_tii'     : X_test_tii,
        'y_test_tii'     : y_test_tii,
        'X_train_base'   : X_train_base,
        'y_train_base'   : y_train_base,
        'X_test_base'    : X_test_base,
        'y_test_base'    : y_test_base,
        'scaler'         : scaler,
        'feature_cols'   : FEATURE_COLS,
        'feature_cols_tii': FEATURE_COLS_TII,
        'sequence_len'   : SEQUENCE_LEN,
    }


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    results = run_preprocessing_pipeline()
