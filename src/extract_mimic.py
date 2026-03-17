"""
================================================================================
CKD PROGRESSION PREDICTION PROJECT
Phase 1: Real MIMIC-IV Data Extraction & Integration

Author      : B.Tech CSE (AIML) Minor Project
Description : Extracts longitudinal features from MIMIC-IV core and hosp modules.
              Replaces synthetic data generator to provide a real-world,
              research-grade clinical dataset. Handles variable visit histories.

Clinical Basis:
  - KDIGO 2012 staging guidelines
  - Validated CKD-EPI equation for eGFR calculation
  - ICD-9/ICD-10 mappings for Comorbidities
================================================================================
"""

import os
import gc
import pandas as pd
import numpy as np

MIMIC_DIR = r"C:\Users\Vanshika Selvam\Downloads"
OUTPUT_FILE = "data/ckd_longitudinal.csv"
MIN_VISITS = 3  # Time-series needs at least 3 longitudinal points

# Exact ItemIDs from MIMIC-IV d_labitems
LAB_ITEMIDS = {
    'creatinine': [50912],
    'hba1c': [50868],
    'hemoglobin': [51222, 50811]
}

def load_demographics():
    print("Loading Demographics (patients.csv.gz)...")
    patients = pd.read_csv(os.path.join(MIMIC_DIR, 'patients.csv.gz'), 
                           usecols=['subject_id', 'gender', 'anchor_age', 'anchor_year'])
    return patients

def load_admissions():
    print("Loading Admissions (admissions.csv.gz)...")
    admissions = pd.read_csv(os.path.join(MIMIC_DIR, 'admissions.csv.gz'),
                             usecols=['subject_id', 'hadm_id', 'admittime'])
    admissions['admittime'] = pd.to_datetime(admissions['admittime'])
    # Extract year to calculate approximate age at admission
    admissions['admin_year'] = admissions['admittime'].dt.year
    return admissions

def load_comorbidities():
    print("Loading Comorbidities (diagnoses_icd.csv.gz)...")
    diagnoses = pd.read_csv(os.path.join(MIMIC_DIR, 'diagnoses_icd.csv.gz'),
                            usecols=['subject_id', 'icd_code', 'icd_version'])
    
    # ICD-9 and ICD-10 prefix mapping using vectorization for speed and memory efficiency
    diagnoses['icd_code'] = diagnoses['icd_code'].astype(str)
    
    # Diabetes
    dia_9 = (diagnoses['icd_version'] == 9) & diagnoses['icd_code'].str.startswith('250')
    dia_10 = (diagnoses['icd_version'] == 10) & diagnoses['icd_code'].str.slice(0, 3).isin(['E08', 'E09', 'E10', 'E11', 'E13'])
    diagnoses['has_diabetes'] = (dia_9 | dia_10).astype(int)
    
    # Hypertension
    hyp_9 = (diagnoses['icd_version'] == 9) & (diagnoses['icd_code'].str.startswith('401') | 
                                               diagnoses['icd_code'].str.startswith('402') |
                                               diagnoses['icd_code'].str.startswith('403') |
                                               diagnoses['icd_code'].str.startswith('404') |
                                               diagnoses['icd_code'].str.startswith('405'))
    hyp_10 = (diagnoses['icd_version'] == 10) & diagnoses['icd_code'].str.slice(0, 3).isin(['I10', 'I11', 'I12', 'I13', 'I15'])
    diagnoses['has_hypertension'] = (hyp_9 | hyp_10).astype(int)
    
    # Aggregate to subject level (has they ever had these diagnosis?)
    comorb = diagnoses.groupby('subject_id').agg({
        'has_diabetes': 'max',
        'has_hypertension': 'max'
    }).reset_index()
    
    del diagnoses
    gc.collect()
    return comorb

def load_omr_vitals():
    print("Loading Vitals (omr.csv.gz)...")
    try:
        omr = pd.read_csv(os.path.join(MIMIC_DIR, 'omr.csv.gz'))
        # Filter for Blood Pressure
        bp_records = omr[omr['result_name'].str.contains('Blood Pressure', case=False, na=False)].copy()
        
        # OMR BP is typically format '120/80'
        bp_records[['systolic_bp', 'diastolic_bp']] = bp_records['result_value'].str.split('/', expand=True)
        bp_records['systolic_bp'] = pd.to_numeric(bp_records['systolic_bp'], errors='coerce')
        bp_records['diastolic_bp'] = pd.to_numeric(bp_records['diastolic_bp'], errors='coerce')
        
        bp_records['chartdate'] = pd.to_datetime(bp_records['chartdate'])
        
        # Group by subject and date (mean if multiple on same date)
        vitals = bp_records.groupby(['subject_id', 'chartdate']).agg({
            'systolic_bp': 'mean',
            'diastolic_bp': 'mean'
        }).reset_index()
        
        del omr, bp_records
        gc.collect()
        return vitals
    except FileNotFoundError:
        print("Warning: omr.csv.gz not found. Will leave blood pressure blank to be backfilled.")
        return pd.DataFrame(columns=['subject_id', 'chartdate', 'systolic_bp', 'diastolic_bp'])

def load_labevents_chunked():
    print("Loading Labs in chunks (labevents.csv.gz)...")
    target_items = set()
    for ids in LAB_ITEMIDS.values():
        target_items.update(ids)
        
    chunk_size = 5000000 
    lab_list = []
    
    # We only care about subject, hadm_id/charttime, itemid, valuenum
    cols = ['subject_id', 'hadm_id', 'itemid', 'charttime', 'valuenum']
    
    for i, chunk in enumerate(pd.read_csv(os.path.join(MIMIC_DIR, 'labevents.csv.gz'), usecols=cols, chunksize=chunk_size)):
        filtered = chunk[chunk['itemid'].isin(target_items)]
        filtered = filtered.dropna(subset=['valuenum'])
        lab_list.append(filtered)
        print(f"  Processed {chunk_size * (i+1):,} rows... Found {len(filtered):,} relevant labs in chunk.")
        
    all_labs = pd.concat(lab_list, ignore_index=True)
    all_labs['charttime'] = pd.to_datetime(all_labs['charttime'])
    
    # Map itemids to standard names
    all_labs['lab_name'] = all_labs['itemid'].map(
        lambda x: 'creatinine' if x in LAB_ITEMIDS['creatinine'] else 
                  'hba1c' if x in LAB_ITEMIDS['hba1c'] else 'hemoglobin'
    )
    
    # Pivot labs into columns, grouping by hadm_id (or charttime date if hadm_id missing)
    all_labs['date'] = all_labs['charttime'].dt.date
    
    # Aggregate labs for each subject and date
    pivoted_labs = all_labs.pivot_table(
        index=['subject_id', 'date'],
        columns='lab_name',
        values='valuenum',
        aggfunc='mean'
    ).reset_index()
    
    pivoted_labs['date'] = pd.to_datetime(pivoted_labs['date'])
    return pivoted_labs

def calculate_egfr(creatinine, age, gender):
    """
    Calculate eGFR using the CKD-EPI 2021 equation (without race).
    Scr: serum creatinine (mg/dL)
    Age: years
    Gender: 'M' or 'F'
    """
    if pd.isna(creatinine) or pd.isna(age) or creatinine <= 0:
        return np.nan
        
    kappa = 0.7 if gender == 'F' else 0.9
    alpha = -0.241 if gender == 'F' else -0.302
    sex_factor = 1.012 if gender == 'F' else 1.0
    
    min_factor = min(creatinine / kappa, 1) ** alpha
    max_factor = max(creatinine / kappa, 1) ** -1.200
    
    egfr = 142 * min_factor * max_factor * (0.9938 ** age) * sex_factor
    return max(1.0, round(egfr, 2))  # Ensure eGFR > 0

def egfr_to_stage(egfr):
    if pd.isna(egfr): return np.nan
    if egfr >= 90: return 1
    elif egfr >= 60: return 2
    elif egfr >= 30: return 3
    elif egfr >= 15: return 4
    else: return 5

def main():
    print("=" * 65)
    print("  MIMIC-IV Dynamic Extraction Script")
    print("=" * 65)
    
    patients = load_demographics()
    admissions = load_admissions()
    comorb = load_comorbidities()
    vitals = load_omr_vitals()
    labs = load_labevents_chunked()
    
    # Merge Demographics and Admissions
    df = pd.merge(admissions, patients, on='subject_id', how='left')
    
    # Compute age at admission
    df['age'] = df['anchor_age'] + (df['admin_year'] - df['anchor_year'])
    df['visit_date'] = df['admittime']
    
    # Merge Comorbidities
    df = pd.merge(df, comorb, on='subject_id', how='left')
    df['has_diabetes'] = df['has_diabetes'].fillna(False).astype(int)
    df['has_hypertension'] = df['has_hypertension'].fillna(False).astype(int)
    
    # Merge Labs & Vitals using Date approximation (within 1 day)
    print("Merging Vitals and Labs into longitudinal visits...")
    df['date_key'] = df['visit_date'].dt.date
    vitals['date_key'] = vitals['chartdate'].dt.date
    labs['date_key'] = labs['date'].dt.date
    
    df = pd.merge(df, vitals.groupby(['subject_id', 'date_key']).mean().reset_index(), 
                  on=['subject_id', 'date_key'], how='left')
    df = pd.merge(df, labs.groupby(['subject_id', 'date_key']).mean().reset_index(), 
                  on=['subject_id', 'date_key'], how='left')
    
    # We only care about visits that have Creatinine (critical for CKD tracking)
    # If creatinine is missing, we drop the 'visit' as it isn't useful for CKD tracking
    df = df.dropna(subset=['creatinine'])
    
    # Calculate eGFR and CKD Stage
    print("Calculating eGFR and Staging...")
    df['egfr'] = df.apply(lambda row: calculate_egfr(row['creatinine'], row['age'], row['gender']), axis=1)
    df['ckd_stage'] = df['egfr'].apply(egfr_to_stage)
    
    # Sort chronologically per patient
    df = df.sort_values(['subject_id', 'visit_date']).reset_index(drop=True)
    
    # Assign visit_number dynamically per patient
    df['visit_number'] = df.groupby('subject_id').cumcount() + 1
    
    # Filter for patients with at least MIN_VISITS
    print(f"Filtering for patients with >= {MIN_VISITS} visits...")
    visit_counts = df.groupby('subject_id').size()
    valid_patients = visit_counts[visit_counts >= MIN_VISITS].index
    df = df[df['subject_id'].isin(valid_patients)].copy()
    
    # Generate progression_label: 1 if Stage increases in NEXT visit, else 0
    # For the last visit, we assume 0 (or we could drop it as a target, but we'll use 0 for now)
    df['next_stage'] = df.groupby('subject_id')['ckd_stage'].shift(-1)
    df['progression_label'] = ((df['next_stage'] > df['ckd_stage']) & df['next_stage'].notna()).astype(int)
    
    # Format and rename columns to match existing pipeline
    df = df.rename(columns={'subject_id': 'patient_id'})
    df['patient_id'] = 'P' + df['patient_id'].astype(str).str.zfill(6)
    
    # Select final columns
    final_cols = [
        'patient_id', 'visit_number', 'visit_date', 'age', 'gender',
        'has_diabetes', 'has_hypertension', 'egfr', 'creatinine',
        'systolic_bp', 'diastolic_bp', 'hba1c', 'hemoglobin',
        'ckd_stage', 'progression_label'
    ]
    
    # Ensure all columns exist even if some are completely missing
    for col in final_cols:
        if col not in df.columns:
            df[col] = np.nan
            
    df = df[final_cols]
    
    # Ensure directory exists and save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    
    print("\n✅ MIMIC-IV Dataset Extraction Complete!")
    print(f"   Shape           : {df.shape}")
    print(f"   Total Patients  : {df['patient_id'].nunique()}")
    print(f"   Longest History : {df.groupby('patient_id').size().max()} visits")
    print(f"   Progression Rate: {df['progression_label'].mean():.1%}")
    print(f"   Saved → {OUTPUT_FILE}")
    print("=" * 65)

if __name__ == '__main__':
    main()
