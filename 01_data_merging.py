import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================
# Path Setup
# ============================================
DATA_PATH = "archive/"          # raw data folder
OUTPUT_PATH = "data/"

os.makedirs(OUTPUT_PATH, exist_ok=True)

print("="*60)
print("STEP 1: Data Merging Started")
print("="*60)

# ============================================
# 1. Load Files (exact filenames)
# ============================================
print("\n[1] Loading CSV files...")

try:
    train = pd.read_csv(os.path.join(DATA_PATH, "Train-1542865627584.csv"))
    beneficiary = pd.read_csv(os.path.join(DATA_PATH, "Train_Beneficiarydata-1542865627584.csv"))
    inpatient = pd.read_csv(os.path.join(DATA_PATH, "Train_Inpatientdata-1542865627584.csv"))
    outpatient = pd.read_csv(os.path.join(DATA_PATH, "Train_Outpatientdata-1542865627584.csv"))
    
    print("✓ All files loaded successfully!")
except Exception as e:
    print("❌ Error while loading files:")
    print(e)
    print("\nCheck:")
    print("1. archive/ folder e files ache kina")
    print("2. File names exact match kore kina")
    exit()

print("\nDataset Shapes:")
print(f"Train (labels)     : {train.shape}")
print(f"Beneficiary        : {beneficiary.shape}")
print(f"Inpatient          : {inpatient.shape}")
print(f"Outpatient         : {outpatient.shape}")

# ============================================
# 2. Add ClaimType and Align Columns
# ============================================
print("\n[2] Preparing Inpatient + Outpatient...")

inpatient['ClaimType'] = 'Inpatient'
outpatient['ClaimType'] = 'Outpatient'

# Common columns find kora
common_cols = list(set(inpatient.columns) & set(outpatient.columns))
print(f"Common columns found: {len(common_cols)}")

# Only common columns niye concat (safe way)
inpatient_common = inpatient[common_cols]
outpatient_common = outpatient[common_cols]

claims = pd.concat([inpatient_common, outpatient_common], ignore_index=True)
print(f"✓ Combined Claims shape: {claims.shape}")

# ============================================
# 3. Merge Beneficiary
# ============================================
print("\n[3] Merging Beneficiary data...")

claims = claims.merge(beneficiary, on='BeneID', how='left')
print(f"✓ After Beneficiary merge: {claims.shape}")

# ============================================
# 4. Merge Fraud Label (Provider level)
# ============================================
print("\n[4] Merging Fraud Labels...")

final_df = claims.merge(train, on='Provider', how='left')
print(f"✓ Final Merged shape: {final_df.shape}")

# ============================================
# 5. Check Fraud Distribution
# ============================================
print("\n[5] Fraud Distribution (PotentialFraud):")
print(final_df['PotentialFraud'].value_counts(dropna=False))
print("\nMissing PotentialFraud:", final_df['PotentialFraud'].isnull().sum())

# ============================================
# 6. Save
# ============================================
output_file = os.path.join(OUTPUT_PATH, "merged_healthcare_data.csv")
final_df.to_csv(output_file, index=False)

print("\n" + "="*60)
print(f"✓ Merged data saved to → {output_file}")
print("="*60)
print("\nStep 1 Completed Successfully!")