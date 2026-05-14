
import pandas as pd
import numpy as np

np.random.seed(42)

# ─── Configuration ───────────────────────────────────────────────
N_SAMPLES = 5000
YEARS = list(range(2010, 2026))  # 16 years of data
SEASONS = ['Kharif', 'Rabi', 'Zaid']  # Indian crop seasons
CROPS = ['Rice', 'Wheat', 'Maize', 'Soybean']
SOIL_TYPES = ['Clay', 'Loam', 'Sandy', 'Silt']
LOCATIONS = ['Region_North', 'Region_South', 'Region_East', 'Region_West', 'Region_Central']

# ─── Generate Base Features ──────────────────────────────────────
data = {
    # 🕐 Time & Location
    'Year': np.random.choice(YEARS, N_SAMPLES),
    'Season': np.random.choice(SEASONS, N_SAMPLES),
    'Location': np.random.choice(LOCATIONS, N_SAMPLES),

    # 🌾 Crop Info
    'Crop_Type': np.random.choice(CROPS, N_SAMPLES),

    # 🌦️ Weather
    'Temperature_C': np.random.normal(27, 5, N_SAMPLES),
    'Rainfall_mm': np.random.normal(800, 300, N_SAMPLES),
    'Humidity_pct': np.random.normal(70, 15, N_SAMPLES),

    # 🌱 Soil
    'Soil_Type': np.random.choice(SOIL_TYPES, N_SAMPLES),
    'Soil_Moisture_pct': np.random.normal(40, 15, N_SAMPLES),

    # 🧪 Management (Controllable by farmer)
    'Irrigation_mm': np.random.normal(300, 150, N_SAMPLES),
    'N_Fertilizer': np.random.normal(120, 40, N_SAMPLES),
    'P_Fertilizer': np.random.normal(40, 20, N_SAMPLES),
    'K_Fertilizer': np.random.normal(40, 20, N_SAMPLES),
}

df = pd.DataFrame(data)

# ─── Clip Negative Values ────────────────────────────────────────
for col in ['Rainfall_mm', 'Humidity_pct', 'Soil_Moisture_pct',
            'Irrigation_mm', 'N_Fertilizer', 'P_Fertilizer', 'K_Fertilizer']:
    df[col] = df[col].clip(lower=0)
df['Humidity_pct'] = df['Humidity_pct'].clip(upper=100)
df['Soil_Moisture_pct'] = df['Soil_Moisture_pct'].clip(upper=100)

# ─── Season-based Temperature Adjustment ─────────────────────────
season_temp_adj = {'Kharif': 3, 'Rabi': -4, 'Zaid': 5}
df['Temperature_C'] = df.apply(
    lambda r: r['Temperature_C'] + season_temp_adj.get(r['Season'], 0), axis=1
)

# ─── Yield Calculation (Agronomical Rules) ────────────────────────
def calculate_yield(row):
    # Base yield by crop (kg/ha)
    base = {'Rice': 3000, 'Wheat': 2500, 'Maize': 4000, 'Soybean': 1500}[row['Crop_Type']]

    # -- Water Factor (Rainfall + Irrigation, diminishing returns) --
    total_water = row['Rainfall_mm'] + row['Irrigation_mm']
    optimal_water = {'Rice': 1200, 'Wheat': 600, 'Maize': 800, 'Soybean': 500}[row['Crop_Type']]
    water_factor = 1.0 - (abs(total_water - optimal_water) / optimal_water) ** 1.5
    water_factor = max(0.2, min(1.2, water_factor))

    # -- Nitrogen Factor --
    optimal_n = {'Rice': 150, 'Wheat': 120, 'Maize': 180, 'Soybean': 40}[row['Crop_Type']]
    n_factor = 1.0 - (abs(row['N_Fertilizer'] - optimal_n) / (optimal_n + 1)) ** 1.5
    n_factor = max(0.4, min(1.1, n_factor))

    # -- Soil Factor --
    soil_bonus = {'Clay': 0.95, 'Loam': 1.10, 'Sandy': 0.85, 'Silt': 1.00}[row['Soil_Type']]
    moisture_factor = 1.0 - abs(row['Soil_Moisture_pct'] - 45) / 100
    moisture_factor = max(0.5, moisture_factor)

    # -- Humidity Factor --
    humidity_factor = 1.0 - abs(row['Humidity_pct'] - 65) / 150
    humidity_factor = max(0.6, humidity_factor)

    # -- Season Factor --
    season_crop = {
        ('Rice', 'Kharif'): 1.15, ('Rice', 'Rabi'): 0.80, ('Rice', 'Zaid'): 0.90,
        ('Wheat', 'Kharif'): 0.75, ('Wheat', 'Rabi'): 1.20, ('Wheat', 'Zaid'): 0.85,
        ('Maize', 'Kharif'): 1.10, ('Maize', 'Rabi'): 0.90, ('Maize', 'Zaid'): 1.00,
        ('Soybean', 'Kharif'): 1.10, ('Soybean', 'Rabi'): 0.85, ('Soybean', 'Zaid'): 0.95,
    }
    season_factor = season_crop.get((row['Crop_Type'], row['Season']), 1.0)

    # -- Year Trend (slight improvement over time) --
    year_factor = 1.0 + (row['Year'] - 2010) * 0.005

    # -- Random Noise (pests, unseen factors) --
    noise = np.random.normal(1.0, 0.08)

    final = base * water_factor * n_factor * soil_bonus * moisture_factor * humidity_factor * season_factor * year_factor * noise
    return max(0, final)

df['Yield_kg_per_ha'] = df.apply(calculate_yield, axis=1)
df = df.round(2)

# ─── Save ──────────────────────────────────────────────────────
output_path = 'data/FarmTwin_Dataset_v2.csv'
df.to_csv(output_path, index=False)

print(f"Generated {N_SAMPLES} records with {len(df.columns)} features!")
print(f"Years: {min(YEARS)} - {max(YEARS)}")
print(f"Columns: {list(df.columns)}")
print(f"Saved to: {output_path}")
print(f"\n--- Sample Data ---")
print(df.head())
print(f"\n--- Train/Test Split Preview ---")
train = df[df['Year'] < 2022]
test = df[df['Year'] >= 2022]
print(f"Train (< 2022): {len(train)} rows")
print(f"Test  (≥ 2022): {len(test)} rows")
