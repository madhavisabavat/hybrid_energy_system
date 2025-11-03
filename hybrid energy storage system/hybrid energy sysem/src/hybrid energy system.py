# =====================================================
# Simple Raw Data Preprocessing (Colab)
# =====================================================

!pip install pandas numpy scikit-learn

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from google.colab import files

# 1️⃣ Upload your raw CSV
print("📤 Please upload your raw_data.csv file...")
uploaded = files.upload()

for fn in uploaded.keys():
    raw_path = fn

# 2️⃣ Load CSV
data = pd.read_csv(raw_path)
print("\n✅ Raw data loaded!")
print("Columns:", data.columns)
display(data.head())

# 3️⃣ Basic cleaning
data = data.drop_duplicates()
data = data.fillna(method='ffill')

# 4️⃣ Feature engineering (adjust column names to match your CSV)
# Replace these with your actual column names if different
# Example columns: Voltage, Current, Temperature, Electrode_Area, Hydrogen_Output
try:
    data['Current_Density'] = data['Current'] / data['Electrode_Area']
    data['Power'] = data['Voltage'] * data['Current']
    data['Efficiency'] = (data['Hydrogen_Output'] / data['Power']) * 100
except KeyError:
    print("⚠️ Some expected columns not found. Skipping feature engineering.")

# 5️⃣ Normalize numeric columns
numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
scaler = StandardScaler()
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

# 6️⃣ Save processed CSV
processed_filename = "processed_data.csv"
data.to_csv(processed_filename, index=False)
print(f"\n💾 Processed data saved as '{processed_filename}'")

# 7️⃣ Download the processed CSV
files.download(processed_filename)
