import pandas as pd
import numpy as np

# Seed for reproducibility
np.random.seed(42)

# Parameters
n_samples = 2000
crops = ['Rice', 'Wheat', 'Maize', 'Soybean']

# Generate base distributions
data = {
    'Crop_Type': np.random.choice(crops, n_samples),
    'Temperature_C': np.random.normal(25, 5, n_samples),
    'Rainfall_mm': np.random.normal(800, 300, n_samples),
    'Irrigation_mm': np.random.normal(300, 150, n_samples),  # Controllable 1
    'N_Fertilizer': np.random.normal(120, 40, n_samples),    # Controllable 2 (Nitrogen)
    'P_Fertilizer': np.random.normal(40, 20, n_samples),     # Phosphorous
    'K_Fertilizer': np.random.normal(40, 20, n_samples),     # Potassium
}

df = pd.DataFrame(data)

# Clip negative values
for col in ['Rainfall_mm', 'Irrigation_mm', 'N_Fertilizer', 'P_Fertilizer', 'K_Fertilizer']:
    df[col] = df[col].clip(lower=0)

# Calculate Yield using an agronomical rule-based approach (with diminishing returns & noise)
def calculate_yield(row):
    # Base yield logic by crop
    base_yield = {'Rice': 3000, 'Wheat': 2500, 'Maize': 4000, 'Soybean': 1500}[row['Crop_Type']]
    
    # Total water = Rainfall + Irrigation
    total_water = row['Rainfall_mm'] + row['Irrigation_mm']
    
    # Water Factor (Diminishing returns + Penalty for flooding/drought)
    optimal_water = {'Rice': 1200, 'Wheat': 600, 'Maize': 800, 'Soybean': 500}[row['Crop_Type']]
    water_factor = 1.0 - (abs(total_water - optimal_water) / optimal_water)**2
    water_factor = max(0.2, water_factor) # Don't drop below 20%
    
    # Fertilizer Factor (Diminishing returns, Liebig's Law of Minimum proxy)
    # Using Nitrogen as primary driver for simulation clarity
    optimal_n = {'Rice': 150, 'Wheat': 120, 'Maize': 180, 'Soybean': 40}[row['Crop_Type']]
    n_factor = 1.0 - (abs(row['N_Fertilizer'] - optimal_n) / (optimal_n + 1))**2
    n_factor = max(0.4, n_factor)
    
    # Combined Multiplier + random noise (weather, pests unseen)
    noise = np.random.normal(1.0, 0.1)
    
    final_yield = base_yield * water_factor * n_factor * noise
    return max(0, final_yield)

df['Yield_kg_per_ha'] = df.apply(calculate_yield, axis=1)

# Round values
df = df.round(2)

# Save
file_path = 'FarmTwin_Yield_Dataset.csv'
df.to_csv(file_path, index=False)
print(f"✅ Generated {n_samples} realistic agricultural simulation records!")
print(f"Saved to: {file_path}")
print(df.head())
