"""
FarmTwin v2 — Climate Trends Engine
Analyzes historical climate data to produce data-driven future projections.

Data Sources:
  - India Rainfall 1901-2015 (115 years, 36 subdivisions)
  - IPCC AR6 warming rates (RCP scenarios)
  - FarmTwin Dataset v2 (local variability calibration)

Methodology:
  1. Fit linear trend on historical rainfall per region
  2. Extract residuals as natural variability distribution
  3. Use IPCC-informed temperature warming rates
  4. Monte Carlo simulation → confidence bands (P10/P50/P90)
"""
import numpy as np
import pandas as pd
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
# 1. REGION → SUBDIVISION MAPPING
# ═══════════════════════════════════════════════════════════════════

# Map FarmTwin regions to representative Indian Meteorological subdivisions
# from the "rainfall in india 1901-2015.csv" dataset.
REGION_SUBDIVISION_MAP = {
    'Region_North':   ['PUNJAB', 'HARYANA DELHI & CHANDIGARH', 'WEST UTTAR PRADESH'],
    'Region_South':   ['TAMIL NADU', 'COASTAL ANDHRA PRADESH', 'KERALA'],
    'Region_East':    ['GANGETIC WEST BENGAL', 'BIHAR', 'JHARKHAND'],
    'Region_West':    ['WEST RAJASTHAN', 'GUJARAT REGION', 'SAURASHTRA & KUTCH'],
    'Region_Central': ['EAST MADHYA PRADESH', 'VIDARBHA', 'CHHATTISGARH'],
}

# IPCC AR6-based warming rates (°C per year) for different RCP scenarios.
# Reference: IPCC AR6 WG1, Chapter 4, Table 4.5
IPCC_WARMING_RATES = {
    'optimistic':  0.015,   # SSP1-2.6: ~1.5°C by 2100 → ~0.015°C/yr
    'moderate':    0.030,   # SSP2-4.5: ~2.7°C by 2100 → ~0.030°C/yr
    'pessimistic': 0.055,   # SSP5-8.5: ~4.4°C by 2100 → ~0.055°C/yr
}

SCENARIO_LABELS = {
    'optimistic':  'Optimistic (SSP1-2.6)',
    'moderate':    'Moderate (SSP2-4.5)',
    'pessimistic': 'Pessimistic (SSP5-8.5)',
}


# ═══════════════════════════════════════════════════════════════════
# 2. HISTORICAL RAINFALL TREND ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def _load_rainfall_history(data_dir='data'):
    """Load the 1901-2015 rainfall dataset."""
    path = Path(data_dir) / 'rainfall in india 1901-2015.csv'
    df = pd.read_csv(path)
    # Clean: ensure ANNUAL is numeric
    df['ANNUAL'] = pd.to_numeric(df['ANNUAL'], errors='coerce')
    return df


def fit_rainfall_trend(region, data_dir='data'):
    """
    Fit a linear trend on historical annual rainfall for the given FarmTwin region.

    Uses subdivision mapping to aggregate rainfall from relevant Indian Met subdivisions.
    Returns: dict with slope (mm/year), intercept, residual_std, r_squared, n_years, subdivisions_used
    """
    rainfall_df = _load_rainfall_history(data_dir)
    subdivisions = REGION_SUBDIVISION_MAP.get(region, REGION_SUBDIVISION_MAP['Region_Central'])

    # Filter to relevant subdivisions and compute regional mean per year
    mask = rainfall_df['SUBDIVISION'].isin(subdivisions)
    regional = rainfall_df[mask].groupby('YEAR')['ANNUAL'].mean().dropna()

    if len(regional) < 20:
        # Fallback: use all subdivisions
        regional = rainfall_df.groupby('YEAR')['ANNUAL'].mean().dropna()

    years = regional.index.values.astype(float)
    rainfall = regional.values

    # Fit linear regression: rainfall = slope * year + intercept
    coeffs = np.polyfit(years, rainfall, 1)
    slope, intercept = coeffs

    # Compute residuals for variability estimation
    predicted = np.polyval(coeffs, years)
    residuals = rainfall - predicted
    residual_std = np.std(residuals)

    # R-squared
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((rainfall - np.mean(rainfall)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return {
        'slope': slope,                   # mm per year (positive = wetter, negative = drier)
        'intercept': intercept,
        'residual_std': residual_std,      # natural year-to-year variability (mm)
        'r_squared': r_squared,
        'n_years': len(years),
        'mean_rainfall': np.mean(rainfall),
        'subdivisions_used': subdivisions,
        'slope_pct_per_year': (slope / np.mean(rainfall)) * 100,  # slope as % of mean
    }


def fit_temp_trend(region, climate_scenario='moderate', data_dir='data'):
    """
    Compute temperature trend using IPCC warming rate + local variability from FarmTwin data.

    The warming rate comes from IPCC AR6; the variability is calibrated from the
    FarmTwin dataset's year-to-year temperature fluctuations for the given region.
    """
    warming_rate = IPCC_WARMING_RATES.get(climate_scenario, IPCC_WARMING_RATES['moderate'])

    # Estimate local temperature variability from FarmTwin dataset
    try:
        farm_df = pd.read_csv(Path(data_dir) / 'FarmTwin_Dataset_v2.csv')
        region_data = farm_df[farm_df['Location'] == region]
        yearly_temp = region_data.groupby('Year')['Temperature_C'].mean()
        if len(yearly_temp) > 2:
            # Detrend and compute std of residuals
            years = yearly_temp.index.values.astype(float)
            temps = yearly_temp.values
            coeffs = np.polyfit(years, temps, 1)
            residuals = temps - np.polyval(coeffs, years)
            temp_variability = np.std(residuals)
        else:
            temp_variability = 0.3  # fallback
    except Exception:
        temp_variability = 0.3  # fallback

    return {
        'warming_rate': warming_rate,            # °C per year
        'variability_std': temp_variability,     # natural variability std
        'scenario': climate_scenario,
        'scenario_label': SCENARIO_LABELS.get(climate_scenario, climate_scenario),
    }


# ═══════════════════════════════════════════════════════════════════
# 3. MONTE CARLO FUTURE CLIMATE PROJECTIONS
# ═══════════════════════════════════════════════════════════════════

def generate_climate_projections(region, years_ahead, climate_scenario='moderate',
                                  n_simulations=100, data_dir='data'):
    """
    Generate Monte Carlo climate projections for future years.

    For each simulation:
      - Temperature: base + warming_rate * y + N(0, variability)
      - Rainfall: base * (1 + trend_pct * y / 100) + N(0, residual_std)

    Returns: dict with:
      - 'projections': list of n_simulations DataFrames (year, temp_change, rain_change_pct)
      - 'rainfall_trend': dict from fit_rainfall_trend
      - 'temp_trend': dict from fit_temp_trend
      - 'summary': DataFrame with Year, P10/P50/P90 for temp and rain changes
    """
    rain_trend = fit_rainfall_trend(region, data_dir)
    temp_trend = fit_temp_trend(region, climate_scenario, data_dir)

    # Pre-compute all random samples at once for efficiency
    np.random.seed(None)  # ensure randomness
    temp_noise = np.random.normal(0, temp_trend['variability_std'], (n_simulations, years_ahead))
    rain_noise = np.random.normal(0, rain_trend['residual_std'], (n_simulations, years_ahead))

    # Temperature changes per year per simulation
    # temp_change[sim, year] = warming_rate * year + noise
    year_indices = np.arange(years_ahead)
    temp_changes = temp_trend['warming_rate'] * year_indices[np.newaxis, :] + temp_noise

    # Rainfall changes as percentage of mean
    # rain_change_mm = slope * year + noise → convert to percentage
    rain_changes_mm = rain_trend['slope'] * year_indices[np.newaxis, :] + rain_noise
    rain_changes_pct = (rain_changes_mm / rain_trend['mean_rainfall']) * 100

    # Build summary statistics
    summary_rows = []
    for y in range(years_ahead):
        temp_vals = temp_changes[:, y]
        rain_vals = rain_changes_pct[:, y]
        summary_rows.append({
            'Year_Offset': y,
            'Temp_Change_P10': np.percentile(temp_vals, 10),
            'Temp_Change_P50': np.percentile(temp_vals, 50),
            'Temp_Change_P90': np.percentile(temp_vals, 90),
            'Rain_Change_Pct_P10': np.percentile(rain_vals, 10),
            'Rain_Change_Pct_P50': np.percentile(rain_vals, 50),
            'Rain_Change_Pct_P90': np.percentile(rain_vals, 90),
        })

    return {
        'temp_changes': temp_changes,          # shape: (n_simulations, years_ahead)
        'rain_changes_pct': rain_changes_pct,  # shape: (n_simulations, years_ahead)
        'rainfall_trend': rain_trend,
        'temp_trend': temp_trend,
        'summary': pd.DataFrame(summary_rows),
    }


def get_trend_description(region, climate_scenario='moderate', data_dir='data'):
    """Generate a human-readable summary of the climate trends for a region."""
    rain = fit_rainfall_trend(region, data_dir)
    temp = fit_temp_trend(region, climate_scenario, data_dir)

    rain_direction = "increasing" if rain['slope'] > 0 else "decreasing"
    rain_abs = abs(rain['slope'])

    return (
        f"📊 **Data Sources**: India Meteorological Dept. rainfall records (1901–2015, "
        f"{rain['n_years']} years) + IPCC AR6 warming projections\n\n"
        f"🌧️ **Rainfall Trend** ({', '.join(rain['subdivisions_used'])}): "
        f"{rain_direction} at **{rain_abs:.2f} mm/year** "
        f"({rain['slope_pct_per_year']:+.3f}%/year of mean {rain['mean_rainfall']:.0f} mm). "
        f"Year-to-year variability: ±{rain['residual_std']:.0f} mm (1σ). "
        f"R² = {rain['r_squared']:.3f}\n\n"
        f"🌡️ **Temperature Trend** ({temp['scenario_label']}): "
        f"+{temp['warming_rate']:.3f}°C/year. "
        f"Local variability: ±{temp['variability_std']:.2f}°C (1σ)\n\n"
        f"🎲 **Method**: Monte Carlo simulation (100 runs) → P10/P50/P90 confidence bands"
    )
