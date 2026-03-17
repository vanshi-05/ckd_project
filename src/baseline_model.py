"""
================================================================================
CKD PROGRESSION PREDICTION PROJECT
Phase 3: Baseline Last-Visit Model

Author      : B.Tech CSE (AIML) Minor Project
Description : Trains a Random Forest classifier on the MOST RECENT VISIT ONLY
              for each patient. This is the baseline that represents current
              single-timepoint clinical decision support tools.

              Deliberately ignores temporal history → establishes lower bound
              that LSTM must beat to validate our novelty claim.

Models compared within baseline:
  1. Logistic Regression (interpretable benchmark)
  2. Random Forest       (primary baseline — selected by AUC)
  3. XGBoost (GradientBoosting) (state-of-art tabular baseline)

Outputs:
  models/baseline_rf_model.pkl     → Best baseline model
  models/baseline_lr_model.pkl     → Logistic regression variant
  plots/baseline_*.png             → Evaluation plots
  results/baseline_results.csv     → Metrics table
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
import os

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report, average_precision_score,
    precision_recall_curve
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
os.makedirs('models',  exist_ok=True)
os.makedirs('plots',   exist_ok=True)
os.makedirs('results', exist_ok=True)

FEATURE_COLS = [
    'egfr', 'creatinine', 'systolic_bp', 'diastolic_bp',
    'hba1c', 'hemoglobin', 'age', 'gender_encoded'
]
RANDOM_STATE = 42


# ─────────────────────────────────────────────────────────────
# HELPER: Evaluate model and return metrics dict
# ─────────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """Compute all evaluation metrics for a trained model."""
    y_pred      = model.predict(X_test)
    y_prob      = model.predict_proba(X_test)[:, 1]

    metrics = {
        'model'     : model_name,
        'accuracy'  : accuracy_score(y_test, y_pred),
        'precision' : precision_score(y_test, y_pred, zero_division=0),
        'recall'    : recall_score(y_test, y_pred, zero_division=0),
        'f1'        : f1_score(y_test, y_pred, zero_division=0),
        'auc_roc'   : roc_auc_score(y_test, y_prob),
        'avg_prec'  : average_precision_score(y_test, y_prob),
    }
    return metrics, y_pred, y_prob


# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
def load_baseline_data():
    """Load the preprocessed baseline (last-visit) feature arrays."""
    print("[BASELINE] Loading preprocessed baseline data...")
    data = np.load('data/ckd_baseline.npz')
    X_train = data['X_train']
    y_train = data['y_train']
    X_test  = data['X_test']
    y_test  = data['y_test']

    print(f"  Train: {X_train.shape}, Progression rate: {y_train.mean():.1%}")
    print(f"  Test : {X_test.shape},  Progression rate: {y_test.mean():.1%}")

    # Compute class weights (for handling imbalance)
    classes      = np.array([0.0, 1.0])
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    cw_dict      = {0: class_weights[0], 1: class_weights[1]}
    print(f"  Class weights: {cw_dict}")

    return X_train, y_train, X_test, y_test, cw_dict


# ─────────────────────────────────────────────────────────────
# TRAIN ALL BASELINE MODELS
# ─────────────────────────────────────────────────────────────
def train_baseline_models(X_train, y_train, cw_dict):
    """
    Train 3 baseline models and compare via 5-fold cross-validation.
    All models use ONLY the last visit's features.
    """
    print("\n[BASELINE] Training models...")

    models = {
        'Logistic Regression': LogisticRegression(
            class_weight='balanced',
            max_iter=500,
            random_state=RANDOM_STATE,
            C=0.1
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=RANDOM_STATE
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {}

    for name, model in models.items():
        print(f"\n  Training {name}...")
        model.fit(X_train, y_train)
        cv_scores = cross_val_score(model, X_train, y_train,
                                    cv=cv, scoring='roc_auc', n_jobs=-1)
        cv_results[name] = {
            'model'   : model,
            'cv_auc'  : cv_scores.mean(),
            'cv_std'  : cv_scores.std(),
        }
        print(f"    5-Fold CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return models, cv_results


# ─────────────────────────────────────────────────────────────
# GENERATE EVALUATION PLOTS
# ─────────────────────────────────────────────────────────────
def generate_baseline_plots(model, X_test, y_test, feature_cols):
    """Generate all evaluation plots for the best baseline model."""

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # ── ROC Curve ──
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color='#3498db', lw=2.5,
            label=f'Random Forest (AUC = {auc_score:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--',
            linewidth=1.5, label='Random Classifier (AUC = 0.50)')
    ax.fill_between(fpr, tpr, alpha=0.08, color='#3498db')
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title('ROC Curve — Baseline Last-Visit Model\n(Random Forest)',
                 fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig('plots/baseline_roc_curve.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("  → Saved: plots/baseline_roc_curve.png")

    # ── Precision-Recall Curve (more informative for imbalanced) ──
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    ap_score = average_precision_score(y_test, y_prob)
    baseline_prec = y_test.mean()

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color='#e74c3c', lw=2.5,
            label=f'Random Forest (AP = {ap_score:.4f})')
    ax.axhline(baseline_prec, color='gray', linestyle='--',
               label=f'No-skill baseline ({baseline_prec:.3f})')
    ax.fill_between(recall, precision, alpha=0.08, color='#e74c3c')
    ax.set_xlabel('Recall (Sensitivity)')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve — Baseline Model\n(More informative under class imbalance)',
                 fontweight='bold')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('plots/baseline_pr_curve.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("  → Saved: plots/baseline_pr_curve.png")

    # ── Confusion Matrix ──
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Progression\n(Pred)', 'Progression\n(Pred)'],
                yticklabels=['No Progression\n(True)', 'Progression\n(True)'],
                ax=ax, cbar_kws={'label': 'Count'},
                linewidths=0.5)
    ax.set_title('Confusion Matrix — Baseline Random Forest', fontweight='bold')

    # Add metric annotations
    tn, fp, fn, tp = cm.ravel()
    ax.text(1.15, 0.5,
            f'Sensitivity: {tp/(tp+fn+1e-9):.3f}\n'
            f'Specificity: {tn/(tn+fp+1e-9):.3f}\n'
            f'PPV: {tp/(tp+fp+1e-9):.3f}\n'
            f'NPV: {tn/(tn+fn+1e-9):.3f}',
            transform=ax.transAxes, va='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='#f0f4ff', edgecolor='#c0cce0'))
    plt.tight_layout()
    plt.savefig('plots/baseline_confusion_matrix.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("  → Saved: plots/baseline_confusion_matrix.png")

    # ── Feature Importance ──
    importances = model.feature_importances_
    feature_names = [
        'eGFR', 'Creatinine', 'Systolic BP', 'Diastolic BP',
        'HbA1c', 'Hemoglobin', 'Age', 'Gender'
    ]
    sorted_idx = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#e74c3c' if i == sorted_idx[0] else '#3498db' for i in range(len(feature_names))]
    bars = ax.bar([feature_names[i] for i in sorted_idx],
                  [importances[i] for i in sorted_idx],
                  color=['#e74c3c' if i == 0 else '#3498db' for i in range(len(sorted_idx))],
                  edgecolor='white', linewidth=1.2)

    # Percentage labels
    for bar, val in zip(bars, [importances[i] for i in sorted_idx]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_title('Feature Importance — Baseline Random Forest\n(Gini Impurity Reduction)',
                 fontweight='bold')
    ax.set_ylabel('Feature Importance Score')
    ax.set_xlabel('Clinical Feature')
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig('plots/baseline_shap.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("  → Saved: plots/baseline_shap.png  (Feature Importance)")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def run_baseline(save_model_path='models/baseline_rf_model.pkl'):
    print("=" * 65)
    print("  Phase 3: Baseline Last-Visit Model Training")
    print("=" * 65)

    X_train, y_train, X_test, y_test, cw_dict = load_baseline_data()
    models, cv_results = train_baseline_models(X_train, y_train, cw_dict)

    # ── Evaluate all on test set ──
    print("\n[BASELINE] Test Set Evaluation:")
    print("-" * 65)
    all_metrics = []
    for name, model in models.items():
        metrics, _, _ = evaluate_model(model, X_test, y_test, name)
        all_metrics.append(metrics)
        print(f"  {name:<25} | AUC: {metrics['auc_roc']:.4f} | "
              f"F1: {metrics['f1']:.4f} | Recall: {metrics['recall']:.4f}")

    # ── Select best model (by AUC-ROC) ──
    best_name = max(all_metrics, key=lambda x: x['auc_roc'])['model']
    best_model = models[best_name]
    print(f"\n  ★ Best baseline model: {best_name}")

    # Save
    rf_model = models['Random Forest']   # Always save RF as primary baseline
    joblib.dump(rf_model, save_model_path)
    joblib.dump(models['Logistic Regression'], 'models/baseline_lr_model.pkl')
    print(f"\n  Saved: {save_model_path}")

    # ── Generate Plots ──
    print("\n[BASELINE] Generating evaluation plots...")
    generate_baseline_plots(rf_model, X_test, y_test, FEATURE_COLS)

    # ── Print full classification report ──
    y_pred_rf = rf_model.predict(X_test)
    print("\n[BASELINE] Full Classification Report (Random Forest):")
    print(classification_report(y_test, y_pred_rf,
                                target_names=['No Progression', 'Progression']))

    # ── Save metrics to CSV ──
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df = metrics_df.round(4)
    metrics_df.to_csv('results/baseline_results.csv', index=False)
    print(f"\n  Metrics saved: results/baseline_results.csv")

    # ── Highlight model limitations (for report) ──
    print("\n[BASELINE] Known Limitations of Last-Visit Approach:")
    print("  ✗ Cannot detect gradual eGFR decline trends")
    print("  ✗ Treats eGFR=45 (stable 3 visits) same as eGFR=45 (dropped from 65)")
    print("  ✗ No temporal context → misses rate-of-change signals")
    print("  ✗ Sensitive to single-visit lab outliers")
    print("  → These limitations motivate the LSTM model in Phase 4")

    print("\n" + "=" * 65)
    print("  ✅ Phase 3 Complete")
    print("=" * 65)

    # Return metrics for comparison in Phase 5
    rf_metrics, rf_pred, rf_prob = evaluate_model(rf_model, X_test, y_test, 'Random Forest')
    return rf_model, rf_metrics, rf_pred, rf_prob, X_test, y_test


if __name__ == '__main__':
    run_baseline()
