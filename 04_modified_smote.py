import pandas as pd
import numpy as np
from imblearn.over_sampling import BorderlineSMOTE
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("STEP 4: Modified SMOTE (Borderline-SMOTE)")
print("="*60)

# ============================================
# 1. Load Noise-Filtered Data
# ============================================
print("\n[1] Loading noise-filtered data...")
df = pd.read_csv("data/cleaned_no_noise_data.csv")
print(f"✓ Data loaded. Shape: {df.shape}")

# ============================================
# 2. Separate Features and Target
# ============================================
print("\n[2] Separating features and target...")

X = df.drop('PotentialFraud', axis=1)
y = df['PotentialFraud']

print("Original class distribution:")
print(Counter(y))
print(y.value_counts(normalize=True).round(3) * 100)

# ============================================
# 3. Apply Borderline-SMOTE (Modified SMOTE)
# ============================================
print("\n[3] Applying Borderline-SMOTE...")
print("This may take 1-2 minutes. Please wait...")

# Borderline-SMOTE parameters
smote = BorderlineSMOTE(
    sampling_strategy='auto',      # minority কে majority এর সমান করবে
    k_neighbors=5,
    kind='borderline-1',           # borderline-1 সবচেয়ে বেশি ব্যবহৃত
    random_state=42
)

X_resampled, y_resampled = smote.fit_resample(X, y)

print("✓ Borderline-SMOTE completed!")

# ============================================
# 4. Check New Distribution
# ============================================
print("\n[4] Class distribution after Modified SMOTE:")
print(Counter(y_resampled))
print(pd.Series(y_resampled).value_counts(normalize=True).round(3) * 100)

print(f"\nOriginal shape   : {X.shape}")
print(f"After SMOTE shape: {X_resampled.shape}")

# ============================================
# 5. Save Resampled Data
# ============================================
resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
resampled_df['PotentialFraud'] = y_resampled

output_file = "data/resampled_data_smote.csv"
resampled_df.to_csv(output_file, index=False)

print("\n" + "="*60)
print(f"✓ Resampled data saved to → {output_file}")
print("="*60)
print("\nStep 4 Completed Successfully!")