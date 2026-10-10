import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ==============================================================================
# STEP 1: Simulate Raw Logistics Dataset with Data Quality Issues
# ==============================================================================
np.random.seed(42)
n_records = 150

raw_data = {
    'Shipment_ID': [f'SHP_{1000+i}' for i in range(n_records)],
    'Distance_KM': np.random.uniform(10, 500, n_records),
    'Package_Weight_KG': np.random.uniform(0.5, 30.0, n_records),
    'Traffic_Density': np.random.choice(['Low', 'Medium', 'High', None], n_records, p=[0.4, 0.3, 0.2, 0.1]),
    'Delay_Minutes': np.random.uniform(0, 180, n_records)
}

df = pd.DataFrame(raw_data)

# Injecting missing values and extreme outliers
df.loc[10:15, 'Distance_KM'] = np.nan
df.loc[25:27, 'Delay_Minutes'] = 1200.0  # Outliers

print("--- Raw Data Missing Values Count ---")
print(df.isnull().sum())

# ==============================================================================
# STEP 2: Data Cleaning - Handling Missing Values
# ==============================================================================
# Numerical Imputation using Median
median_distance = df['Distance_KM'].median()
df['Distance_KM'].fillna(median_distance, inplace=True)

# Categorical Imputation using Mode
mode_traffic = df['Traffic_Density'].mode()[0]
df['Traffic_Density'].fillna(mode_traffic, inplace=True)

# ==============================================================================
# STEP 3: Outlier Detection and Capping using IQR Method
# ==============================================================================
Q1 = df['Delay_Minutes'].quantile(0.25)
Q3 = df['Delay_Minutes'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Capping outliers
df['Delay_Minutes_Cleaned'] = np.where(df['Delay_Minutes'] > upper_bound, upper_bound, df['Delay_Minutes'])

# ==============================================================================
# STEP 4: Feature Normalization (MinMaxScaler)
# ==============================================================================
scaler = MinMaxScaler()
df[['Distance_Scaled', 'Weight_Scaled']] = scaler.fit_transform(df[['Distance_KM', 'Package_Weight_KG']])

print("\n--- Preprocessed Data Sample ---")
print(df[['Shipment_ID', 'Distance_Scaled', 'Weight_Scaled', 'Traffic_Density', 'Delay_Minutes_Cleaned']].head())
