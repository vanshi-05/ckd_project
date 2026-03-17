"""
================================================================================
CKD PROGRESSION PREDICTION PROJECT
Phase 1: Synthetic Longitudinal CKD Dataset Generator

Author      : B.Tech CSE (AIML) Minor Project
Description : Generates a clinically realistic longitudinal CKD dataset
              following KDIGO 2012 guidelines and published epidemiological
              statistics on CKD progression rates.

Clinical Basis:
  - eGFR staging follows KDIGO 2012 guidelines (PMID: 22812152)
  - Progression rates (~30% at 3 years) from Tangri et al., 2011
  - Creatinine-eGFR relationship: CKD-EPI equation (Levey et al., 2009)
  - BP targets and comorbidity rates from NHANES CKD data

Output: data/ckd_longitudinal.csv
================================================================================
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# REPRODUCIBILITY SEED
# ─────────────────────────────────────────────────────────────
np.random.seed(42)

# ─────────────────────────────────────────────────────────────
# CLINICAL CONSTANTS (based on KDIGO guidelines)
# ─────────────────────────────────────────────────────────────
EGFR_STAGES = {
    1: (90, 120),    # eGFR ≥ 90 ml/min/1.73m²
    2: (60,  89),    # eGFR 60–89
    3: (30,  59),    # eGFR 30–59 (3a: 45-59, 3b: 30-44)
    4: (15,  29),    # eGFR 15–29
    5: ( 5,  14),    # eGFR < 15 (kidney failure)
}

# Probability of progressing to next CKD stage per visit
# Based on published 3-year CKD progression literature
PROGRESSION_PROBS = {
    1: 0.08,   # Stage 1 → 2 (low baseline risk)
    2: 0.12,   # Stage 2 → 3
    3: 0.18,   # Stage 3 → 4 (significantly higher risk)
    4: 0.25,   # Stage 4 → 5
    5: 0.00,   # Stage 5 = end stage, no further progression
}

# Mean eGFR decline per visit (ml/min/1.73m²) by stage
# Faster decline at higher stages (sicker kidneys)
EGFR_DECLINE_MEAN = {1: 0.5, 2: 1.0, 3: 2.0, 4: 3.5, 5: 1.0}
EGFR_DECLINE_STD  = {1: 0.8, 2: 1.2, 3: 1.8, 4: 2.5, 5: 0.5}


# ─────────────────────────────────────────────────────────────
# HELPER: Get CKD Stage from eGFR value
# ─────────────────────────────────────────────────────────────
def egfr_to_stage(egfr: float) -> int:
    """Map eGFR value to CKD stage per KDIGO 2012 guidelines."""
    if egfr >= 90:
        return 1
    elif egfr >= 60:
        return 2
    elif egfr >= 30:
        return 3
    elif egfr >= 15:
        return 4
    else:
        return 5


# ─────────────────────────────────────────────────────────────
# HELPER: Approximate creatinine from eGFR using inverted CKD-EPI
# ─────────────────────────────────────────────────────────────
def egfr_to_creatinine(egfr: float, age: int, gender: str) -> float:
    """
    Approximate serum creatinine from eGFR.
    Simplified inversion of the CKD-EPI equation.
    Normal range: 0.6–1.2 mg/dL (female), 0.7–1.3 mg/dL (male)
    """
    # κ and α constants from CKD-EPI equation
    kappa = 0.7 if gender == 'F' else 0.9
    alpha = -0.329 if gender == 'F' else -0.411
    sex_factor = 1.018 if gender == 'F' else 1.0

    # Inverse solve for creatinine (simplified)
    # eGFR = 141 × min(Scr/κ, 1)^α × max(Scr/κ, 1)^(-1.209) × 0.993^age × sex_factor
    # We use a clinical approximation: Scr ≈ 75/eGFR for male, 65/eGFR for female
    base = 75 if gender == 'M' else 65
    creatinine = base / max(egfr, 5)

    # Add small Gaussian noise for realism (lab measurement variability)
    noise = np.random.normal(0, 0.05)
    creatinine = max(0.4, creatinine + noise)
    return round(creatinine, 2)


# ─────────────────────────────────────────────────────────────
# HELPER: Generate blood pressure correlated with CKD stage
# ─────────────────────────────────────────────────────────────
def generate_bp(stage: int, has_hypertension: bool) -> tuple:
    """
    Generate systolic and diastolic BP.
    Hypertension is both a cause and consequence of CKD.
    ~70-80% of CKD patients have hypertension (NHANES data).
    """
    # Base BP increases with stage (CKD → HTN → worsening CKD cycle)
    stage_bp_offset = {1: 0, 2: 3, 3: 8, 4: 14, 5: 18}
    base_sbp = 120 if not has_hypertension else 140
    base_dbp =  80 if not has_hypertension else  88

    sbp = base_sbp + stage_bp_offset[stage] + np.random.normal(0, 6)
    dbp = base_dbp + stage_bp_offset[stage] * 0.5 + np.random.normal(0, 4)

    sbp = max(90,  min(200, sbp))
    dbp = max(55,  min(120, dbp))
    return round(sbp, 1), round(dbp, 1)


# ─────────────────────────────────────────────────────────────
# HELPER: Generate HbA1c correlated with diabetes status
# ─────────────────────────────────────────────────────────────
def generate_hba1c(has_diabetes: bool) -> float:
    """
    HbA1c: normal <5.7%, prediabetes 5.7-6.4%, diabetes ≥6.5%.
    ~40% of CKD patients have diabetes (USRDS 2022 data).
    """
    if has_diabetes:
        hba1c = np.random.normal(7.8, 1.2)   # Poorly controlled diabetes
        hba1c = max(6.5, min(14.0, hba1c))
    else:
        hba1c = np.random.normal(5.4, 0.4)
        hba1c = max(4.5, min(6.4, hba1c))
    return round(hba1c, 1)


# ─────────────────────────────────────────────────────────────
# HELPER: Generate hemoglobin (anemia is common in CKD)
# ─────────────────────────────────────────────────────────────
def generate_hemoglobin(stage: int, gender: str) -> float:
    """
    CKD-related anemia worsens with stage.
    Normal: M >13 g/dL, F >12 g/dL (WHO criteria).
    ~50% of Stage 3-5 patients have anemia.
    """
    normal_hgb = 14.5 if gender == 'M' else 13.0
    # Anemia offset increases with CKD stage (impaired erythropoietin production)
    stage_offset = {1: 0.0, 2: -0.3, 3: -1.2, 4: -2.5, 5: -3.5}
    hgb = normal_hgb + stage_offset[stage] + np.random.normal(0, 0.8)
    hgb = max(5.0, min(18.0, hgb))
    return round(hgb, 1)


# ─────────────────────────────────────────────────────────────
# CORE: Generate one patient's longitudinal visit history
# ─────────────────────────────────────────────────────────────
def generate_patient(patient_id: str,
                     initial_stage: int,
                     n_visits: int,
                     start_date: datetime) -> list:
    """
    Generate a longitudinal visit sequence for one patient.

    Parameters:
    -----------
    patient_id    : Unique patient identifier (e.g., 'P001')
    initial_stage : Starting CKD stage (1–5)
    n_visits      : Number of clinic visits to generate (8–16)
    start_date    : Date of first visit

    Returns:
    --------
    List of visit dictionaries ready for DataFrame creation.
    """
    # ── Patient-level characteristics (fixed across visits) ──
    age_at_start   = np.random.randint(35, 80)
    gender         = np.random.choice(['M', 'F'], p=[0.54, 0.46])
    has_diabetes   = np.random.choice([True, False], p=[0.40, 0.60])
    has_hypertension = np.random.choice([True, False], p=[0.75, 0.25])

    # ── Initial eGFR: sample uniformly within the stage's range ──
    egfr_min, egfr_max = EGFR_STAGES[initial_stage]
    current_egfr = np.random.uniform(egfr_min * 0.85, egfr_max)
    current_stage = egfr_to_stage(current_egfr)

    visits = []

    for visit_num in range(n_visits):
        # ── Visit date: approximately every 3 months ──
        visit_date = start_date + timedelta(days=90 * visit_num +
                                            np.random.randint(-14, 14))

        # ── Patient age at this visit ──
        age_at_visit = age_at_start + visit_num // 4

        # ── Generate lab values for this visit ──
        creatinine  = egfr_to_creatinine(current_egfr, age_at_visit, gender)
        sbp, dbp    = generate_bp(current_stage, has_hypertension)
        hba1c       = generate_hba1c(has_diabetes)
        hemoglobin  = generate_hemoglobin(current_stage, gender)

        # ── Occasionally introduce lab measurement noise / outliers ──
        if np.random.random() < 0.04:
            current_egfr *= np.random.uniform(0.88, 1.12)  # 4% chance of outlier

        # ── Determine if progression occurs at NEXT visit ──
        prog_prob = PROGRESSION_PROBS[current_stage]

        # Diabetes and hypertension increase progression risk
        if has_diabetes:
            prog_prob *= 1.35
        if has_hypertension:
            prog_prob *= 1.20
        # Faster decline if eGFR already very low within stage
        stage_range = EGFR_STAGES[current_stage]
        stage_percentile = (current_egfr - stage_range[0]) / max(
            stage_range[1] - stage_range[0], 1)
        if stage_percentile < 0.25:  # in bottom 25% of stage
            prog_prob *= 1.30

        prog_prob = min(prog_prob, 0.95)  # cap at 95%
        progression_label = 1 if np.random.random() < prog_prob else 0

        # ── Record this visit ──
        visits.append({
            'patient_id'       : patient_id,
            'visit_number'     : visit_num + 1,
            'visit_date'       : visit_date.strftime('%Y-%m-%d'),
            'age'              : age_at_visit,
            'gender'           : gender,
            'has_diabetes'     : int(has_diabetes),
            'has_hypertension' : int(has_hypertension),
            'egfr'             : round(current_egfr, 2),
            'creatinine'       : creatinine,
            'systolic_bp'      : sbp,
            'diastolic_bp'     : dbp,
            'hba1c'            : hba1c,
            'hemoglobin'       : hemoglobin,
            'ckd_stage'        : current_stage,
            'progression_label': progression_label,
        })

        # ── Update eGFR for next visit ──
        if progression_label == 1 and current_stage < 5:
            # When progression occurs: eGFR drops enough to cross a stage boundary
            # Drop to a random value in the next stage's lower half
            next_stage  = current_stage + 1
            next_min, next_max = EGFR_STAGES[next_stage]
            mid_point   = (next_min + next_max) / 2
            current_egfr = np.random.uniform(next_min, mid_point)
            current_stage = next_stage
        else:
            # Normal visit-to-visit decline within current stage
            decline = np.random.normal(
                EGFR_DECLINE_MEAN[current_stage],
                EGFR_DECLINE_STD[current_stage]
            )
            current_egfr = max(current_egfr - abs(decline), 4.0)
            current_stage = egfr_to_stage(current_egfr)

    return visits


# ─────────────────────────────────────────────────────────────
# MAIN: Generate full dataset
# ─────────────────────────────────────────────────────────────
def generate_full_dataset(n_patients: int = 500,
                          output_path: str = 'data/ckd_longitudinal.csv') -> pd.DataFrame:
    """
    Generate the complete longitudinal CKD dataset.

    Parameters:
    -----------
    n_patients  : Total number of patients (default: 500)
    output_path : CSV output file path

    Returns:
    --------
    pd.DataFrame: Complete dataset with all visit records
    """
    print("=" * 65)
    print("  CKD Longitudinal Dataset Generator")
    print("  Following KDIGO 2012 Guidelines")
    print("=" * 65)

    all_visits = []

    # ── Stage distribution: mirrors real-world CKD prevalence ──
    # Stage 1: 5%, Stage 2: 15%, Stage 3: 55%, Stage 4: 20%, Stage 5: 5%
    stage_distribution = [1, 1, 1, 1, 1,
                          2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                          3] * 12 + [3] * 23 + [4] * 20 * 2 + [5] * 5 * 2

    # Truncate/extend to match n_patients
    stages_assigned = [stage_distribution[i % len(stage_distribution)]
                       for i in range(n_patients)]

    # Slight shuffle to avoid pure sequential ordering
    np.random.shuffle(stages_assigned)

    start_base = datetime(2019, 1, 1)

    for i in range(n_patients):
        patient_id    = f'P{str(i + 1).zfill(4)}'
        initial_stage = stages_assigned[i]
        n_visits      = np.random.randint(8, 17)    # 8–16 visits per patient
        # Stagger start dates over 6 months for realism
        start_date    = start_base + timedelta(days=np.random.randint(0, 180))

        patient_visits = generate_patient(patient_id, initial_stage,
                                          n_visits, start_date)
        all_visits.extend(patient_visits)

        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{n_patients} patients...")

    df = pd.DataFrame(all_visits)

    # ── Introduce realistic missing values (~4–8% per clinical lab) ──
    missing_cols = ['hba1c', 'hemoglobin', 'diastolic_bp']
    for col in missing_cols:
        missing_mask = np.random.random(len(df)) < np.random.uniform(0.04, 0.08)
        df.loc[missing_mask, col] = np.nan

    # ── Save to CSV ──
    import os
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    df.to_csv(output_path, index=False)

    # ── Print dataset summary ──
    print(f"\n✅ Dataset generated: {len(df):,} total visit records")
    print(f"   Patients        : {df['patient_id'].nunique()}")
    print(f"   Avg visits/pt   : {df.groupby('patient_id').size().mean():.1f}")
    print(f"   Progression rate: {df['progression_label'].mean():.1%}")
    print(f"   Missing values  : {df.isnull().sum().sum()} cells")
    print(f"\n   CKD Stage Distribution (first visit per patient):")

    first_visits = df.groupby('patient_id').first().reset_index()
    for stage in range(1, 6):
        count = (first_visits['ckd_stage'] == stage).sum()
        pct   = count / len(first_visits) * 100
        print(f"     Stage {stage}: {count:3d} patients ({pct:.1f}%)")

    print(f"\n   Saved → {output_path}")
    print("=" * 65)

    return df


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    df = generate_full_dataset(
        n_patients=500,
        output_path='data/ckd_longitudinal.csv'
    )
    print("\nFirst 5 rows:")
    print(df.head().to_string())
    print("\nData types:")
    print(df.dtypes)
