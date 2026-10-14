import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("STEP 2: Data Preprocessing & Missing Value Handling")
print("="*60)

# ============================================
# 1. Load Merged Data
# ============================================
print("\n[1] Loading merged data...")

df = pd.read_csv("data/merged_healthcare_data.csv")
print(f"✓ Data loaded. Shape: {df.shape}")

# ============================================
# 2. Basic Info
# ============================================
print("\n[2] Basic Information:")
print(f"Total rows      : {df.shape[0]}")
print(f"Total columns   : {df.shape[1]}")
print(f"\nFraud Distribution:")
print(df['PotentialFraud'].value_counts())

# ============================================
# 3. Check Missing Values
# ============================================
print("\n[3] Checking Missing Values...")

missing = df.isnull().sum()
missing_percent = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing_Count': missing,
    'Missing_Percent': missing_percent
})
missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Percent', ascending=False)

print("\nColumns with Missing Values:")
print(missing_df.head(20))

# ============================================
# 4. Drop High Missing / Useless Columns
# ============================================
print("\n[4] Dropping high-missing and useless columns...")

# এই কলামগুলো সাধারণত খুব বেশি missing থাকে বা কাজে লাগে না
cols_to_drop = [
    'ClmProcedureCode_1', 'ClmProcedureCode_2', 'ClmProcedureCode_3',
    'ClmProcedureCode_4', 'ClmProcedureCode_5', 'ClmProcedureCode_6',
    'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8',
    'ClmDiagnosisCode_9', 'ClmDiagnosisCode_10',
    'OperatingPhysician', 'OtherPhysician',
    'ClmAdmitDiagnosisCode'
]

# শুধু যেগুলো আছে সেগুলো ড্রপ করব
cols_to_drop = [col for col in cols_to_drop if col in df.columns]
df.drop(columns=cols_to_drop, inplace=True)
print(f"✓ Dropped columns: {cols_to_drop}")

# ============================================
# 5. Handle Remaining Missing Values
# ============================================
print("\n[5] Handling remaining missing values...")

# Numerical columns → median দিয়ে fill
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in num_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# Categorical / Object columns → 'Unknown' দিয়ে fill
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna('Unknown', inplace=True)

print("✓ Missing values handled")

# ============================================
# 6. Convert Target Variable
# ============================================
print("\n[6] Converting Target Variable...")

df['PotentialFraud'] = df['PotentialFraud'].map({'Yes': 1, 'No': 0})
print("✓ PotentialFraud converted to 1 (Yes) and 0 (No)")
print(df['PotentialFraud'].value_counts())

# ============================================
# 7. Convert Date columns (if exist)
# ============================================
print("\n[7] Converting Date columns...")

date_cols = ['ClaimStartDt', 'ClaimEndDt', 'DOB', 'DOD', 'AdmissionDt', 'DischargeDt']
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

print("✓ Date columns converted")

# ============================================
# 8. Final Check
# ============================================
print("\n[8] Final Check after Preprocessing:")
print(f"Shape after preprocessing : {df.shape}")
print(f"Remaining missing values  : {df.isnull().sum().sum()}")

print("\nData Types:")
print(df.dtypes.value_counts())

# ============================================
# 9. Save Cleaned Data
# ============================================
output_file = "data/cleaned_healthcare_data.csv"
df.to_csv(output_file, index=False)

print("\n" + "="*60)
print(f"✓ Cleaned data saved to → {output_file}")
print("="*60)
print("\nStep 2 Completed Successfully!")