import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("STEP 3: Noise/Outlier Removal using IsolationForest")
print("="*60)

# ============================================
# 1. Load Cleaned Data
# ============================================
print("\n[1] Loading cleaned data...")
df = pd.read_csv("data/cleaned_healthcare_data.csv")
print(f"✓ Data loaded. Shape: {df.shape}")

# ============================================
# 2. Select Important Features
# ============================================
print("\n[2] Selecting important numerical features...")

important_features = [
    'InscClaimAmtReimbursed',
    'DeductibleAmtPaid',
    'IPAnnualReimbursementAmt',
    'IPAnnualDeductibleAmt',
    'OPAnnualReimbursementAmt',
    'OPAnnualDeductibleAmt',
    'NoOfMonths_PartACov',
    'NoOfMonths_PartBCov',
    'ChronicCond_Alzheimer',
    'ChronicCond_Heartfailure',
    'ChronicCond_KidneyDisease',
    'ChronicCond_Cancer',
    'ChronicCond_ObstrPulmonary',
    'ChronicCond_Depression',
    'ChronicCond_Diabetes',
    'ChronicCond_IschemicHeart',
    'ChronicCond_Osteoporasis',
    'ChronicCond_rheumatoidarthritis',
    'ChronicCond_stroke'
]

features = [col for col in important_features if col in df.columns]
print(f"✓ Selected {len(features)} features")

y = df['PotentialFraud'].copy()
X = df[features].copy()

# ============================================
# 3. Handle Missing Values
# ============================================
print("\n[3] Handling missing values...")
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=features)
print(f"✓ Missing values after imputation: {X.isnull().sum().sum()}")

# ============================================
# 4. Scaling
# ============================================
print("\n[4] Scaling data...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("✓ Scaling done")

# ============================================
# 5. IsolationForest for Outlier Detection
# ============================================
print("\n[5] Applying IsolationForest (fast)...")

iso = IsolationForest(
    n_estimators=100,
    contamination=0.05,      # 5% কে outlier ধরবে
    random_state=42,
    n_jobs=-1
)

outlier_pred = iso.fit_predict(X_scaled)

# -1 = outlier/noise,  1 = normal
n_noise = list(outlier_pred).count(-1)
print(f"✓ Total Noise/Outlier points : {n_noise}")
print(f"✓ Percentage of Noise        : {(n_noise / len(outlier_pred)) * 100:.2f}%")

# ============================================
# 6. Remove Noise
# ============================================
print("\n[6] Removing noise points...")

mask = outlier_pred == 1          # শুধু normal points রাখব
X_clean = X[mask].reset_index(drop=True)
y_clean = y[mask].reset_index(drop=True)

print(f"✓ Original shape       : {X.shape}")
print(f"✓ After removing noise : {X_clean.shape}")
print(f"✓ Removed samples      : {len(X) - len(X_clean)}")

print("\nFraud distribution after cleaning:")
print(y_clean.value_counts())
print(y_clean.value_counts(normalize=True).round(3)*100)

# ============================================
# 7. Save
# ============================================
clean_df = X_clean.copy()
clean_df['PotentialFraud'] = y_clean

output_file = "data/cleaned_no_noise_data.csv"
clean_df.to_csv(output_file, index=False)

print("\n" + "="*60)
print(f"✓ Clean data saved to → {output_file}")
print("="*60)
print("\nStep 3 Completed Successfully!")