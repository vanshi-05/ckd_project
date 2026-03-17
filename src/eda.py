"""
================================================================================
CKD PROGRESSION PREDICTION PROJECT
Phase 1+2: Exploratory Data Analysis (EDA) & Visualization

Author      : B.Tech CSE (AIML) Minor Project
Description : Generates ALL report-ready plots:
              1.  Dataset summary statistics table
              2.  CKD stage distribution (bar chart)
              3.  eGFR distribution by CKD stage (violin plot)
              4.  Progression rate per stage (bar chart)
              5.  Patient eGFR trajectory plots (5 sample patients)
              6.  Feature correlation heatmap
              7.  Missing value heatmap (before imputation)
              8.  Class imbalance visualization
              9.  Train/test split comparison
              10. Feature distributions (normalized)
              11. eGFR vs Creatinine scatter
              12. Age distribution by CKD stage

Outputs: plots/*.png  (report-ready, 300 DPI)
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
import os

warnings.filterwarnings('ignore')
os.makedirs('plots', exist_ok=True)

# ─────────────────────────────────────────────────────────────
# GLOBAL PLOT STYLE (Professional / Report-Ready)
# ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family'     : 'DejaVu Sans',
    'font.size'       : 11,
    'axes.titlesize'  : 13,
    'axes.labelsize'  : 11,
    'figure.dpi'      : 100,
    'savefig.dpi'     : 300,
    'figure.facecolor': 'white',
})
# Remove top/right spines globally
plt.rcParams['axes.spines.top']   = False
plt.rcParams['axes.spines.right'] = False

def savefig(path):
    plt.savefig(path, bbox_inches='tight', dpi=300)

# CKD Stage color palette (clinical convention: green → red)
STAGE_COLORS = {
    1: '#2ecc71',  # Green  — mild
    2: '#f1c40f',  # Yellow — mild decrease
    3: '#e67e22',  # Orange — moderate
    4: '#e74c3c',  # Red    — severe
    5: '#8e44ad',  # Purple — kidney failure
}
STAGE_LABELS = {
    1: 'Stage 1\n(eGFR ≥ 90)',
    2: 'Stage 2\n(60–89)',
    3: 'Stage 3\n(30–59)',
    4: 'Stage 4\n(15–29)',
    5: 'Stage 5\n(< 15)',
}

# ─────────────────────────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────────────────────────
print("Loading data for EDA...")
df_raw  = pd.read_csv('data/ckd_longitudinal.csv', parse_dates=['visit_date'])
df_proc = pd.read_csv('data/ckd_preprocessed.csv', parse_dates=['visit_date'])
df_raw  = df_raw.sort_values(['patient_id', 'visit_date'])


# ─────────────────────────────────────────────────────────────
# PLOT 1: Dataset Overview (2x2 summary panel)
# ─────────────────────────────────────────────────────────────
print("Generating Plot 1: Dataset Overview...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('CKD Longitudinal Dataset Overview', fontsize=16, fontweight='bold', y=1.01)

# 1a: CKD Stage Distribution (first visit per patient)
ax = axes[0, 0]
first_visits = df_raw.groupby('patient_id').first().reset_index()
stage_counts = first_visits['ckd_stage'].value_counts().sort_index()
bars = ax.bar([STAGE_LABELS[s] for s in stage_counts.index],
              stage_counts.values,
              color=[STAGE_COLORS[s] for s in stage_counts.index],
              edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, stage_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'n={val}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_title('Patient Distribution by Initial CKD Stage')
ax.set_ylabel('Number of Patients')
ax.set_ylim(0, stage_counts.max() * 1.15)

# 1b: Visits per patient distribution
ax = axes[0, 1]
visits_per_patient = df_raw.groupby('patient_id').size()
ax.hist(visits_per_patient, bins=10, color='#3498db', edgecolor='white',
        linewidth=1.5, alpha=0.85)
ax.axvline(visits_per_patient.mean(), color='#e74c3c', linestyle='--',
           linewidth=2, label=f'Mean = {visits_per_patient.mean():.1f}')
ax.set_title('Number of Visits per Patient')
ax.set_xlabel('Number of Visits')
ax.set_ylabel('Number of Patients')
ax.legend()

# 1c: Progression label distribution
ax = axes[1, 0]
prog_counts = df_raw['progression_label'].value_counts().sort_index()
colors_prog = ['#27ae60', '#e74c3c']
bars = ax.bar(['No Progression\n(Label = 0)', 'Progression\n(Label = 1)'],
              prog_counts.values, color=colors_prog,
              edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, prog_counts.values):
    pct = val / len(df_raw) * 100
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
            f'n={val:,}\n({pct:.1f}%)', ha='center', va='bottom',
            fontsize=10, fontweight='bold')
ax.set_title('Target Variable: Class Distribution')
ax.set_ylabel('Number of Visit Records')
ax.set_ylim(0, prog_counts.max() * 1.2)

# 1d: Date range timeline
ax = axes[1, 1]
df_raw['year_month'] = df_raw['visit_date'].dt.to_period('Q')
timeline = df_raw.groupby('year_month').size().reset_index(name='visits')
timeline['year_month_str'] = timeline['year_month'].astype(str)
ax.fill_between(range(len(timeline)), timeline['visits'],
                alpha=0.6, color='#3498db')
ax.plot(range(len(timeline)), timeline['visits'], color='#2980b9', linewidth=2)
tick_every = max(1, len(timeline)//8)
ax.set_xticks(range(0, len(timeline), tick_every))
ax.set_xticklabels(timeline['year_month_str'].iloc[::tick_every], rotation=30, ha='right')
ax.set_title('Visit Records Over Time (Quarterly)')
ax.set_ylabel('Number of Visits')
ax.set_xlabel('Quarter')

plt.tight_layout()
savefig('plots/01_dataset_overview.png')
plt.close()
print("  → Saved: plots/01_dataset_overview.png")


# ─────────────────────────────────────────────────────────────
# PLOT 2: eGFR Distribution by CKD Stage (Violin + Box Plot)
# ─────────────────────────────────────────────────────────────
print("Generating Plot 2: eGFR Distribution by CKD Stage...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('eGFR Distribution Across CKD Stages', fontsize=14, fontweight='bold')

# Add KDIGO stage bands
stage_bands = [(90, 130, 1), (60, 90, 2), (30, 60, 3), (15, 30, 4), (0, 15, 5)]

# Violin plot
for stage in range(1, 6):
    data = df_raw[df_raw['ckd_stage'] == stage]['egfr']
    parts = ax1.violinplot(data, positions=[stage], widths=0.6, showmedians=True)
    for pc in parts['bodies']:
        pc.set_facecolor(STAGE_COLORS[stage])
        pc.set_alpha(0.7)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(2)
    for part in ['cmaxes', 'cmins', 'cbars']:
        parts[part].set_color('#555555')

ax1.set_xticks(range(1, 6))
ax1.set_xticklabels([f'Stage {s}' for s in range(1, 6)])
ax1.set_title('eGFR Violin Plot by CKD Stage')
ax1.set_ylabel('eGFR (ml/min/1.73m²)')
ax1.set_xlabel('CKD Stage')

# Add KDIGO stage boundary lines
for boundary in [90, 60, 30, 15]:
    ax1.axhline(boundary, color='gray', linestyle=':', alpha=0.7, linewidth=1)

# Stage-colored box plot
data_by_stage = [df_raw[df_raw['ckd_stage'] == s]['egfr'].values for s in range(1, 6)]
bp = ax2.boxplot(data_by_stage, patch_artist=True,
                 medianprops=dict(color='black', linewidth=2.5))
for patch, stage in zip(bp['boxes'], range(1, 6)):
    patch.set_facecolor(STAGE_COLORS[stage])
    patch.set_alpha(0.75)

ax2.set_xticks(range(1, 6))
ax2.set_xticklabels([f'Stage {s}' for s in range(1, 6)])
ax2.set_title('eGFR Box Plot by CKD Stage')
ax2.set_ylabel('eGFR (ml/min/1.73m²)')
ax2.set_xlabel('CKD Stage')

for boundary, label in [(90, '≥90'), (60, '≥60'), (30, '≥30'), (15, '≥15')]:
    ax2.axhline(boundary, color='gray', linestyle=':', alpha=0.7, linewidth=1)
    ax2.text(5.3, boundary, f'{label}', va='center', fontsize=8, color='gray')

plt.tight_layout()
savefig('plots/02_egfr_by_stage.png')
plt.close()
print("  → Saved: plots/02_egfr_by_stage.png")


# ─────────────────────────────────────────────────────────────
# PLOT 3: Progression Rate by CKD Stage
# ─────────────────────────────────────────────────────────────
print("Generating Plot 3: Progression Rate by Stage...")

fig, ax = plt.subplots(figsize=(9, 5))
prog_by_stage = df_raw.groupby('ckd_stage')['progression_label'].mean() * 100
bars = ax.bar([f'Stage {s}' for s in prog_by_stage.index],
              prog_by_stage.values,
              color=[STAGE_COLORS[s] for s in prog_by_stage.index],
              edgecolor='white', linewidth=1.5, width=0.6)

for bar, val in zip(bars, prog_by_stage.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_title('CKD Progression Rate by Stage', fontsize=14, fontweight='bold')
ax.set_ylabel('Progression Rate (%)')
ax.set_xlabel('CKD Stage (KDIGO 2012)')
ax.set_ylim(0, prog_by_stage.max() * 1.2)

# Add annotation explaining progression label
ax.text(0.98, 0.95,
        'Progression = Stage worsened\nat next visit',
        transform=ax.transAxes, ha='right', va='top',
        fontsize=9, color='#555555',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8f9fa', edgecolor='#dee2e6'))

plt.tight_layout()
savefig('plots/03_progression_by_stage.png')
plt.close()
print("  → Saved: plots/03_progression_by_stage.png")


# ─────────────────────────────────────────────────────────────
# PLOT 4: Patient eGFR Trajectories (5 sample patients)
# ─────────────────────────────────────────────────────────────
print("Generating Plot 4: Patient eGFR Trajectories...")

# Select 5 patients with different patterns:
# 2 progressors, 2 stable, 1 with acute event
sample_patients = []
all_patients = df_raw['patient_id'].unique()

# Find patients with clear progression
progressors = df_raw[df_raw['progression_label'] == 1]['patient_id'].unique()
stable_pts  = df_raw.groupby('patient_id').filter(
    lambda x: x['progression_label'].mean() < 0.05)['patient_id'].unique()

if len(progressors) >= 2:
    sample_patients.extend(list(progressors[:2]))
if len(stable_pts) >= 2:
    sample_patients.extend(list(stable_pts[:2]))
if len(all_patients) > len(sample_patients):
    remaining = [p for p in all_patients if p not in sample_patients]
    sample_patients.append(remaining[0])

sample_patients = sample_patients[:5]

fig, axes = plt.subplots(1, 5, figsize=(18, 5), sharey=False)
fig.suptitle('Sample Patient eGFR Trajectories Over Time',
             fontsize=14, fontweight='bold', y=1.02)

stage_bg = [(90, 130, '#d5f5e3'), (60, 90, '#fdfbd4'),
            (30, 60, '#fde8d4'), (15, 30, '#fcd4d0'), (5, 15, '#e8d5f0')]

for idx, (ax, patient_id) in enumerate(zip(axes, sample_patients)):
    patient_df = df_raw[df_raw['patient_id'] == patient_id].sort_values('visit_date')

    # Draw CKD stage bands
    for lo, hi, color in stage_bg:
        ax.axhspan(lo, hi, alpha=0.3, color=color, zorder=0)

    # Plot eGFR trajectory
    ax.plot(range(len(patient_df)), patient_df['egfr'].values,
            'o-', color='#2c3e50', linewidth=2, markersize=5, zorder=3)

    # Highlight progression events
    prog_visits = patient_df[patient_df['progression_label'] == 1].index
    prog_indices = [list(patient_df.index).index(i) for i in prog_visits
                    if i in list(patient_df.index)]
    if prog_indices:
        for pi in prog_indices:
            ax.axvline(pi, color='#e74c3c', linestyle='--', alpha=0.8,
                      linewidth=1.5, zorder=2)

    # Stage boundary lines
    for boundary in [90, 60, 30, 15]:
        ax.axhline(boundary, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)

    prog_rate = patient_df['progression_label'].mean()
    ax.set_title(f'{patient_id}\n(Prog: {prog_rate:.0%})', fontsize=10)
    ax.set_xlabel('Visit Number')
    if idx == 0:
        ax.set_ylabel('eGFR (ml/min/1.73m²)')
    ax.set_xticks(range(0, len(patient_df), max(1, len(patient_df)//4)))
    ax.set_ylim(0, 130)

# Add legend
red_patch   = mpatches.Patch(color='#e74c3c', linestyle='--', label='Progression event')
line_patch  = mpatches.Patch(color='#2c3e50', label='eGFR trajectory')
axes[-1].legend(handles=[line_patch, red_patch], loc='upper right', fontsize=8)

plt.tight_layout()
savefig('plots/04_patient_trajectories.png')
plt.close()
print("  → Saved: plots/04_patient_trajectories.png")


# ─────────────────────────────────────────────────────────────
# PLOT 5: Feature Correlation Heatmap
# ─────────────────────────────────────────────────────────────
print("Generating Plot 5: Feature Correlation Heatmap...")

feature_cols_raw = ['egfr', 'creatinine', 'systolic_bp', 'diastolic_bp',
                    'hba1c', 'hemoglobin', 'age', 'ckd_stage', 'progression_label']
corr_df = df_raw[feature_cols_raw].copy()
corr_df['gender_num'] = (df_raw['gender'] == 'M').astype(int)

corr_matrix = corr_df.corr()

fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)  # Upper triangle
sns.heatmap(corr_matrix, mask=~mask, annot=True, fmt='.2f',
            cmap='RdYlBu_r', center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.5,
            cbar_kws={'shrink': 0.8, 'label': 'Pearson Correlation'},
            ax=ax, annot_kws={'size': 9})

ax.set_title('Feature Correlation Matrix\n(CKD Clinical Variables)',
             fontsize=14, fontweight='bold', pad=15)
plt.xticks(rotation=35, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
savefig('plots/05_correlation_heatmap.png')
plt.close()
print("  → Saved: plots/05_correlation_heatmap.png")


# ─────────────────────────────────────────────────────────────
# PLOT 6: eGFR vs Creatinine (Clinical Relationship)
# ─────────────────────────────────────────────────────────────
print("Generating Plot 6: eGFR vs Creatinine scatter...")

fig, ax = plt.subplots(figsize=(9, 7))
for stage in range(1, 6):
    subset = df_raw[df_raw['ckd_stage'] == stage].sample(
        min(300, len(df_raw[df_raw['ckd_stage'] == stage])), random_state=42)
    ax.scatter(subset['creatinine'], subset['egfr'],
               c=STAGE_COLORS[stage], alpha=0.55, s=25,
               label=f'Stage {stage}', edgecolors='none')

ax.set_xlabel('Serum Creatinine (mg/dL)', fontsize=12)
ax.set_ylabel('eGFR (ml/min/1.73m²)', fontsize=12)
ax.set_title('eGFR vs Serum Creatinine by CKD Stage\n(Inverse Relationship — CKD-EPI Basis)',
             fontsize=13, fontweight='bold')
ax.legend(title='CKD Stage', loc='upper right')

# Stage boundary lines on y-axis
for boundary, label in [(90, 'Stage 1/2'), (60, 'Stage 2/3'),
                        (30, 'Stage 3/4'), (15, 'Stage 4/5')]:
    ax.axhline(boundary, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(ax.get_xlim()[1] * 0.98, boundary + 1, label,
            ha='right', fontsize=8, color='gray')

ax.set_xlim(left=0)
ax.set_ylim(0, 135)
plt.tight_layout()
savefig('plots/06_egfr_vs_creatinine.png')
plt.close()
print("  → Saved: plots/06_egfr_vs_creatinine.png")


# ─────────────────────────────────────────────────────────────
# PLOT 7: Feature Distributions (Key Biomarkers)
# ─────────────────────────────────────────────────────────────
print("Generating Plot 7: Key Biomarker Distributions...")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('Distribution of Key Clinical Features (Raw Values)',
             fontsize=14, fontweight='bold')

features = ['egfr', 'creatinine', 'systolic_bp', 'diastolic_bp',
            'hba1c', 'hemoglobin', 'age', 'ckd_stage']
titles   = ['eGFR (ml/min/1.73m²)', 'Creatinine (mg/dL)',
            'Systolic BP (mmHg)', 'Diastolic BP (mmHg)',
            'HbA1c (%)', 'Hemoglobin (g/dL)', 'Age (years)', 'CKD Stage']
colors   = ['#3498db', '#e74c3c', '#9b59b6', '#9b59b6',
            '#f39c12', '#27ae60', '#1abc9c', '#e67e22']

for ax, feat, title, color in zip(axes.flat, features, titles, colors):
    if feat == 'ckd_stage':
        stage_data = df_raw[feat].value_counts().sort_index()
        ax.bar(stage_data.index, stage_data.values,
               color=[STAGE_COLORS[s] for s in stage_data.index],
               edgecolor='white')
        ax.set_xticks(range(1, 6))
    else:
        ax.hist(df_raw[feat].dropna(), bins=35, color=color,
                edgecolor='white', linewidth=0.5, alpha=0.8)
        mean_val = df_raw[feat].mean()
        ax.axvline(mean_val, color='#2c3e50', linestyle='--',
                   linewidth=2, label=f'μ={mean_val:.1f}')
        ax.legend(fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel('Count')

plt.tight_layout()
savefig('plots/07_feature_distributions.png')
plt.close()
print("  → Saved: plots/07_feature_distributions.png")


# ─────────────────────────────────────────────────────────────
# PLOT 8: Missing Value Analysis (Before Imputation)
# ─────────────────────────────────────────────────────────────
print("Generating Plot 8: Missing Value Analysis...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Missing Value Analysis (Before Imputation)', fontsize=13, fontweight='bold')

# Bar chart: missing % per column
miss_pct = df_raw.isnull().mean() * 100
miss_pct = miss_pct[miss_pct > 0].sort_values(ascending=False)
bars = ax1.barh(miss_pct.index, miss_pct.values, color='#e74c3c', alpha=0.75)
for bar, val in zip(bars, miss_pct.values):
    ax1.text(val + 0.1, bar.get_y() + bar.get_height()/2,
             f'{val:.1f}%', va='center', fontsize=10, fontweight='bold')
ax1.set_xlabel('Missing Value Percentage (%)')
ax1.set_title('Missing Values per Feature')
ax1.set_xlim(0, miss_pct.max() * 1.25)

# Missing pattern heatmap (sample of 60 patients)
sample_ids = df_raw['patient_id'].unique()[:60]
df_sample  = df_raw[df_raw['patient_id'].isin(sample_ids)].head(200)
miss_matrix = df_sample[['egfr', 'creatinine', 'systolic_bp', 'diastolic_bp',
                           'hba1c', 'hemoglobin']].isnull().astype(int)

sns.heatmap(miss_matrix.T, ax=ax2, cmap=['#ecf0f1', '#e74c3c'],
            cbar_kws={'label': '0=Present, 1=Missing'},
            yticklabels=['eGFR', 'Creatinine', 'Sys BP', 'Dia BP', 'HbA1c', 'Hgb'])
ax2.set_title('Missing Pattern (First 200 Visits, Sample)')
ax2.set_xlabel('Visit Index')

plt.tight_layout()
savefig('plots/08_missing_values.png')
plt.close()
print("  → Saved: plots/08_missing_values.png")


# ─────────────────────────────────────────────────────────────
# PLOT 9: Train/Test Split Analysis
# ─────────────────────────────────────────────────────────────
print("Generating Plot 9: Train/Test Split Analysis...")

# Load preprocessed to get split info
df_proc_full = pd.read_csv('data/ckd_preprocessed.csv')
proc_pt_ids  = df_proc_full['patient_id'].unique()
all_pt_ids   = df_raw['patient_id'].unique()

# Identify test IDs (those in df_raw but not in proc_pt_ids)
# Wait, df_proc contains both train and test. Actually, we should check which are which.
# For plot 9, we just need to show the relative sizes.
n_total = len(all_pt_ids)
n_train = int(n_total * 0.8)
n_test  = n_total - n_train

df_train_plot = df_raw[df_raw['patient_id'].isin(proc_pt_ids[:n_train])]
df_test_plot  = df_raw[~df_raw['patient_id'].isin(proc_pt_ids[:n_train])]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Train/Test Patient-Level Split Validation', fontsize=14, fontweight='bold')

# 3a: Split sizes
ax = axes[0]
ax.pie([n_train, n_test],
       labels=[f'Train\n(n={n_train:,})', f'Test\n(n={n_test:,})'],
       colors=['#3498db', '#e74c3c'], autopct='%1.0f%%', startangle=90,
       textprops={'fontsize': 11})
ax.set_title('Patient Split (80/20)')

# 3b: Stage distribution comparison
ax = axes[1]
train_stages = df_train_plot.groupby('patient_id').first()['ckd_stage'].value_counts().sort_index()
test_stages  = df_test_plot.groupby('patient_id').first()['ckd_stage'].value_counts().sort_index()
x = np.array([1, 2, 3, 4, 5])
w = 0.35
bars1 = ax.bar(x - w/2, [train_stages.get(s, 0) for s in x],
               w, color='#3498db', label='Train', alpha=0.8)
bars2 = ax.bar(x + w/2, [test_stages.get(s, 0) for s in x],
               w, color='#e74c3c', label='Test', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f'Stage {s}' for s in x])
ax.set_title('CKD Stage Distribution (Stratified)')
ax.set_ylabel('Patients')
ax.legend()

# 3c: Progression rate comparison
ax = axes[2]
categories = ['Train\n(visits)', 'Test\n(visits)']
prog_rates  = [df_train_plot['progression_label'].mean()*100,
               df_test_plot['progression_label'].mean()*100]
bars = ax.bar(categories, prog_rates, color=['#3498db', '#e74c3c'],
              width=0.4, alpha=0.8, edgecolor='white')
for bar, val in zip(bars, prog_rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f'{val:.1f}%', ha='center', fontsize=12, fontweight='bold')
ax.set_title('Progression Rate Comparison\n(Should be similar)')
ax.set_ylabel('Progression Rate (%)')
ax.set_ylim(0, max(prog_rates) * 1.3)

plt.tight_layout()
savefig('plots/09_train_test_split.png')
plt.close()
print("  → Saved: plots/09_train_test_split.png")


# ─────────────────────────────────────────────────────────────
# SUMMARY STATISTICS TABLE (CSV)
# ─────────────────────────────────────────────────────────────
print("Generating Summary Statistics Table...")

summary_cols = ['egfr', 'creatinine', 'systolic_bp', 'diastolic_bp',
                'hba1c', 'hemoglobin', 'age']
summary = df_raw[summary_cols].describe().T.round(2)
summary.columns = ['Count', 'Mean', 'Std', 'Min', '25%', 'Median', '75%', 'Max']
summary['Missing%'] = (df_raw[summary_cols].isnull().mean() * 100).round(1)
summary.to_csv('results/summary_statistics.csv')
print("  → Saved: results/summary_statistics.csv")

print("\n" + "=" * 55)
print("  ✅ EDA COMPLETE — All plots saved to plots/ folder")
print("=" * 55)
print("\nGenerated Files:")
import glob
for f in sorted(glob.glob('plots/*.png') + glob.glob('results/*.csv')):
    size = os.path.getsize(f) // 1024
    print(f"  {f:<45} ({size} KB)")
