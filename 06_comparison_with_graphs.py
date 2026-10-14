import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             average_precision_score, matthews_corrcoef,
                             roc_auc_score, confusion_matrix)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("COMPARISON: Original vs SMOTE vs Our Full Pipeline")
print("="*60)

# ============================================
# Helper Function: Train Ensemble & Get Metrics
# ============================================
def train_and_evaluate(X, y, name="Model"):
    print(f"\n→ Training on {name}...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    rf = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
    xgb = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, 
                        random_state=42, eval_metric='logloss', n_jobs=-1)
    lgbm = LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, 
                          random_state=42, n_jobs=-1, verbose=-1)
    
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('xgb', xgb), ('lgbm', lgbm)],
        voting='soft', n_jobs=-1
    )
    
    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)
    y_proba = ensemble.predict_proba(X_test)[:, 1]
    
    metrics = {
        'Model': name,
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'AUC-PR': average_precision_score(y_test, y_proba),
        'AUC-ROC': roc_auc_score(y_test, y_proba),
        'MCC': matthews_corrcoef(y_test, y_pred)
    }
    
    print(f"✓ {name} done | F1: {metrics['F1-Score']:.4f} | Recall: {metrics['Recall']:.4f}")
    return metrics, y_test, y_pred

# ============================================
# 1. Prepare Three Datasets
# ============================================
print("\n[1] Preparing datasets...")

# ---- Dataset A: Original Imbalanced (after basic cleaning) ----
df_orig = pd.read_csv("data/cleaned_healthcare_data.csv")

features = [
    'InscClaimAmtReimbursed', 'DeductibleAmtPaid',
    'IPAnnualReimbursementAmt', 'IPAnnualDeductibleAmt',
    'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt',
    'NoOfMonths_PartACov', 'NoOfMonths_PartBCov',
    'ChronicCond_Alzheimer', 'ChronicCond_Heartfailure',
    'ChronicCond_KidneyDisease', 'ChronicCond_Cancer',
    'ChronicCond_ObstrPulmonary', 'ChronicCond_Depression',
    'ChronicCond_Diabetes', 'ChronicCond_IschemicHeart',
    'ChronicCond_Osteoporasis', 'ChronicCond_rheumatoidarthritis',
    'ChronicCond_stroke'
]

features = [f for f in features if f in df_orig.columns]
X_orig = df_orig[features]
y_orig = df_orig['PotentialFraud']

imputer = SimpleImputer(strategy='median')
X_orig = pd.DataFrame(imputer.fit_transform(X_orig), columns=features)

print(f"Original Imbalanced shape: {X_orig.shape} | Fraud %: {y_orig.mean()*100:.1f}%")

# ---- Dataset B: Normal SMOTE (without noise filtering) ----
print("Applying normal SMOTE on original data...")
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_orig, y_orig)
print(f"Normal SMOTE shape: {X_smote.shape}")

# ---- Dataset C: Our Full Pipeline (already done) ----
df_our = pd.read_csv("data/resampled_data_smote.csv")
X_our = df_our.drop('PotentialFraud', axis=1)
y_our = df_our['PotentialFraud']
print(f"Our Pipeline shape: {X_our.shape}")

# ============================================
# 2. Train All Three
# ============================================
print("\n[2] Training models on three setups...")

results = []

# A. Original Imbalanced
m1, yt1, yp1 = train_and_evaluate(X_orig, y_orig, "1. Original Imbalanced")
results.append(m1)

# B. Normal SMOTE
m2, yt2, yp2 = train_and_evaluate(X_smote, y_smote, "2. Normal SMOTE")
results.append(m2)

# C. Our Full Pipeline
m3, yt3, yp3 = train_and_evaluate(X_our, y_our, "3. Our Pipeline (IF + BorderlineSMOTE)")
results.append(m3)

# ============================================
# 3. Create Comparison Table
# ============================================
results_df = pd.DataFrame(results)
print("\n" + "="*70)
print("FINAL COMPARISON TABLE")
print("="*70)
print(results_df.round(4).to_string(index=False))
results_df.to_csv("data/final_comparison.csv", index=False)
print("\n✓ Comparison table saved → data/final_comparison.csv")

# ============================================
# 4. Generate Graphs
# ============================================
print("\n[3] Generating graphs...")

plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics_to_plot = ['Precision', 'Recall', 'F1-Score', 'AUC-PR']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

for idx, metric in enumerate(metrics_to_plot):
    ax = axes[idx // 2, idx % 2]
    bars = ax.bar(results_df['Model'], results_df[metric], color=colors, edgecolor='black')
    ax.set_title(f'{metric} Comparison', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(metric)
    ax.tick_params(axis='x', rotation=15)
    
    # value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)

plt.suptitle('Healthcare Fraud Detection - Performance Comparison', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('data/comparison_metrics.png', dpi=300, bbox_inches='tight')
print("✓ Graph saved → data/comparison_metrics.png")

# Confusion Matrix of Our Best Model
plt.figure(figsize=(6, 5))
cm = confusion_matrix(yt3, yp3)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
plt.title('Confusion Matrix - Our Full Pipeline', fontsize=14, fontweight='bold')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('data/confusion_matrix_our_pipeline.png', dpi=300, bbox_inches='tight')
print("✓ Confusion Matrix saved → data/confusion_matrix_our_pipeline.png")

print("\n" + "="*60)
print("All comparisons and graphs completed successfully!")
print("="*60)
print("\nFiles created:")
print("1. data/final_comparison.csv")
print("2. data/comparison_metrics.png")
print("3. data/confusion_matrix_our_pipeline.png")