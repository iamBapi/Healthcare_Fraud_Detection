import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             precision_score, recall_score, f1_score,
                             average_precision_score, matthews_corrcoef,
                             roc_auc_score)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("STEP 5: Ensemble Classification (RF + XGBoost + LightGBM)")
print("="*60)

# ============================================
# 1. Load Resampled Data
# ============================================
print("\n[1] Loading resampled data...")
df = pd.read_csv("data/resampled_data_smote.csv")
print(f"✓ Data loaded. Shape: {df.shape}")

X = df.drop('PotentialFraud', axis=1)
y = df['PotentialFraud']

print("\nClass distribution:")
print(y.value_counts())

# ============================================
# 2. Train-Test Split
# ============================================
print("\n[2] Splitting data (80% train, 20% test)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.20, 
    random_state=42, 
    stratify=y
)

print(f"Train shape: {X_train.shape}")
print(f"Test shape : {X_test.shape}")

# ============================================
# 3. Define Base Models
# ============================================
print("\n[3] Defining models...")

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

xgb = XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss',
    n_jobs=-1
)

lgbm = LGBMClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

# ============================================
# 4. Create Ensemble (Soft Voting)
# ============================================
print("\n[4] Creating Soft Voting Ensemble...")

ensemble = VotingClassifier(
    estimators=[
        ('rf', rf),
        ('xgb', xgb),
        ('lgbm', lgbm)
    ],
    voting='soft',
    n_jobs=-1
)

# ============================================
# 5. Train Ensemble
# ============================================
print("\n[5] Training Ensemble Model (this may take 2-5 minutes)...")
print("Please wait...")

ensemble.fit(X_train, y_train)
print("✓ Training completed!")

# ============================================
# 6. Predictions
# ============================================
print("\n[6] Making predictions...")

y_pred = ensemble.predict(X_test)
y_pred_proba = ensemble.predict_proba(X_test)[:, 1]

# ============================================
# 7. Evaluation Metrics
# ============================================
print("\n" + "="*60)
print("PERFORMANCE EVALUATION")
print("="*60)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Main Metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_pr = average_precision_score(y_test, y_pred_proba)
mcc = matthews_corrcoef(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_pred_proba)

print("\n" + "-"*40)
print("KEY METRICS SUMMARY")
print("-"*40)
print(f"Precision     : {precision:.4f}")
print(f"Recall        : {recall:.4f}")
print(f"F1-Score      : {f1:.4f}")
print(f"AUC-PR        : {auc_pr:.4f}")
print(f"AUC-ROC       : {auc_roc:.4f}")
print(f"MCC           : {mcc:.4f}")
print("-"*40)

# ============================================
# 8. Save Results
# ============================================
results = {
    'Precision': [precision],
    'Recall': [recall],
    'F1-Score': [f1],
    'AUC-PR': [auc_pr],
    'AUC-ROC': [auc_roc],
    'MCC': [mcc]
}

results_df = pd.DataFrame(results)
results_df.to_csv("data/model_performance.csv", index=False)

print("\n✓ Performance results saved to → data/model_performance.csv")
print("\n" + "="*60)
print("Step 5 Completed Successfully!")
print("="*60)