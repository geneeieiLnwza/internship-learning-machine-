"""
FarmTwin v2 — Full Dashboard (Streamlit)
Multi-tab interactive Digital Twin Agriculture Simulator
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
sys.path.insert(0, '.')

from farmtwin.simulation import simulate, run_all_scenarios, predict_future, PREDEFINED_SCENARIOS
from farmtwin.decision import recommend_fertilizer, recommend_crop, assess_risk
from farmtwin.explainability import get_feature_importance, generate_explanation_text

# ─── Page Config ──────────────────────────────────────────────────
st.set_page_config(page_title="FarmTwin v2", page_icon="F", layout="wide")

st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1a2e; color: white; border-radius: 8px;
        padding: 8px 16px;
        border: none !important;
    }
    .stTabs [aria-selected="true"] { background-color: #16813d; color: white; }
    .stTabs [data-baseweb="tab-highlight"] { display: none !important; }
    .stTabs [data-baseweb="tab-border"] { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ─── Load Models ──────────────────────────────────────────────────
@st.cache_resource
def load_all_models():
    rf = joblib.load('models/random_forest.pkl')
    lr = joblib.load('models/linear_regression.pkl')
    ann = joblib.load('models/neural_network.pkl')
    stacking = joblib.load('models/stacking_meta.pkl')
    encoder = joblib.load('models/encoder.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return rf, lr, ann, stacking, encoder, scaler

try:
    rf_model, lr_model, ann_model, stacking_meta, encoder, scaler = load_all_models()
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.info("Please run `python3 farmtwin/model_layer.py` first.")
    st.stop()

model = rf_model


# ─── Sidebar: Environment Inputs ─────────────────────────────────
st.sidebar.title("Environment")
crop = st.sidebar.selectbox("Crop Type", ['Rice', 'Wheat', 'Maize', 'Soybean'])
season = st.sidebar.selectbox("Season", ['Kharif', 'Rabi', 'Zaid'])
location = st.sidebar.selectbox("Location", ['Region_North', 'Region_South', 'Region_East', 'Region_West', 'Region_Central'])
soil_type = st.sidebar.selectbox("Soil Type", ['Clay', 'Loam', 'Sandy', 'Silt'])

st.sidebar.divider()
st.sidebar.subheader("Weather")
temp = st.sidebar.slider("Temperature (C)", 10.0, 45.0, 27.0, 0.5)
rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 2000.0, 800.0, 10.0)
humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 70.0, 1.0)
soil_moisture = st.sidebar.slider("Soil Moisture (%)", 0.0, 100.0, 40.0, 1.0)

st.sidebar.divider()
st.sidebar.subheader("Farm Management")
irrigation = st.sidebar.slider("Irrigation (mm)", 0.0, 1000.0, 300.0, 10.0)
n_fert = st.sidebar.slider("N Fertilizer (kg/ha)", 0.0, 300.0, 120.0, 5.0)
p_fert = st.sidebar.slider("P Fertilizer (kg/ha)", 0.0, 150.0, 40.0, 5.0)
k_fert = st.sidebar.slider("K Fertilizer (kg/ha)", 0.0, 150.0, 40.0, 5.0)

base_params = {
    'Crop_Type': crop, 'Season': season, 'Location': location, 'Soil_Type': soil_type,
    'Temperature_C': temp, 'Rainfall_mm': rainfall, 'Humidity_pct': humidity,
    'Soil_Moisture_pct': soil_moisture, 'Irrigation_mm': irrigation,
    'N_Fertilizer': n_fert, 'P_Fertilizer': p_fert, 'K_Fertilizer': k_fert, 'Year': 2024
}


# ─── Header ───────────────────────────────────────────────────────
st.title("FarmTwin v2: Digital Twin Agriculture Simulator")
st.caption("AI-powered simulation system for smart farming decisions")


# ─── Tabs ─────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Yield Prediction",
    "What-If Simulation",
    "Scenario Analysis",
    "Future Prediction",
    "Decision Support",
    "Model Comparison",
    "Explainable AI (XAI)"
])


# ═══════════ TAB 1: YIELD PREDICTION ═════════════════════════════
with tab1:
    st.header("Yield Prediction")

    baseline, predicted, _ = simulate(model, encoder, scaler, base_params)

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Yield", f"{predicted:,.0f} kg/ha")
    col2.metric("Temperature", f"{temp} C")
    col3.metric("Total Water", f"{rainfall + irrigation:,.0f} mm")

    st.divider()
    st.subheader("Environment Summary")
    info_df = pd.DataFrame([{k: str(v) for k, v in base_params.items()}]).T
    info_df.columns = ['Value']
    st.dataframe(info_df, width='stretch')


# ═══════════ TAB 2: WHAT-IF SIMULATION ═══════════════════════════
with tab2:
    st.header("What-If Simulation")
    st.info("Adjust the values below to see how yield changes when you modify certain factors.")

    c1, c2 = st.columns(2)
    with c1:
        rain_change = st.slider("Rainfall Change (%)", -80, 80, 0, 5, key='wif_rain')
        irr_change = st.slider("Irrigation Change (%)", -80, 80, 0, 5, key='wif_irr')
    with c2:
        n_change = st.slider("N Fertilizer Change (%)", -80, 80, 0, 5, key='wif_n')
        temp_change = st.slider("Temperature Change (C)", -5.0, 5.0, 0.0, 0.5, key='wif_temp')

    changes = {}
    if rain_change != 0: changes['Rainfall_mm'] = f'{rain_change}%'
    if irr_change != 0: changes['Irrigation_mm'] = f'{irr_change}%'
    if n_change != 0: changes['N_Fertilizer'] = f'{n_change}%'
    if temp_change != 0: changes['Temperature_C'] = temp_change

    if changes:
        base_y, sim_y, diff = simulate(model, encoder, scaler, base_params, changes)
        pct = (diff / (base_y + 1)) * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("Baseline", f"{base_y:,.0f} kg/ha")
        c2.metric("Simulated", f"{sim_y:,.0f} kg/ha", f"{diff:+,.0f}")
        c3.metric("Change", f"{pct:+.1f}%", delta_color="normal")

        chart_data = pd.DataFrame({'Scenario': ['Baseline', 'Simulated'], 'Yield (kg/ha)': [base_y, sim_y]})
        st.bar_chart(chart_data.set_index('Scenario'))
    else:
        st.warning("Please adjust at least one value to see the simulation result.")


# ═══════════ TAB 3: SCENARIO ANALYSIS ════════════════════════════
with tab3:
    st.header("Scenario Analysis")

    results_df = run_all_scenarios(model, encoder, scaler, base_params)
    results_df.columns = ['Scenario', 'Baseline Yield', 'Simulated Yield', 'Difference', 'Change (%)']
    st.dataframe(results_df, width='stretch', hide_index=True)

    chart = results_df[['Scenario', 'Baseline Yield', 'Simulated Yield']].set_index('Scenario')
    st.bar_chart(chart)


# ═══════════ TAB 4: FUTURE PREDICTION ════════════════════════════
with tab4:
    st.header("Future Prediction (Time Simulation)")
    years = st.slider("Number of Years to Predict", 1, 10, 5, key='future_years')

    future_df = predict_future(model, encoder, scaler, base_params, years)
    future_df.columns = ['Year', 'Predicted Yield (kg/ha)', 'Temp Change', 'Rainfall Change']

    st.dataframe(future_df, width='stretch', hide_index=True)
    st.line_chart(future_df.set_index('Year')['Predicted Yield (kg/ha)'])


# ═══════════ TAB 5: DECISION SUPPORT ═════════════════════════════
with tab5:
    st.header("Decision Support")

    d1, d2 = st.columns(2)

    with d1:
        st.subheader("Optimal N Fertilizer")
        fert_rec = recommend_fertilizer(model, encoder, scaler, base_params)
        st.success(fert_rec['advice'])
        st.metric("Recommended N", f"{fert_rec['optimal_N']} kg/ha")
        st.metric("Expected Yield", f"{fert_rec['expected_yield']:,.0f} kg/ha")

        curve = fert_rec['curve_data']
        st.line_chart(curve.set_index('N_Fertilizer')['Predicted_Yield'])

    with d2:
        st.subheader("Crop Recommendation")
        crop_rec = recommend_crop(model, encoder, scaler, base_params)
        st.success(crop_rec['advice'])
        st.dataframe(crop_rec['comparison'], width='stretch', hide_index=True)

    st.divider()
    st.subheader("Risk Assessment")
    risk = assess_risk(model, encoder, scaler, base_params)
    r1, r2, r3 = st.columns(3)
    r1.metric("Risk Level", risk['risk_level'])
    r2.metric("Best Case", f"{risk['best_yield']:,.0f} kg/ha")
    r3.metric("Worst Case", f"{risk['worst_yield']:,.0f} kg/ha")
    st.info(f"Recommendation: {risk['recommendation']}")


# ═══════════ TAB 6: MODEL COMPARISON ═════════════════════════════
with tab6:
    st.header("Model Comparison")
    st.caption("Time-based validation (Train < 2022, Test >= 2022) per Paper 8")

    comparison_data = {
        'Model': ['Baseline (Mean)', 'Linear Regression', 'Random Forest', 'Neural Network (ANN)', 'Stacking (RF+ANN)'],
        'RMSE': [946.54, 530.64, 255.96, 242.97, 261.79],
        'R2': [-0.0125, 0.6818, 0.9260, 0.9333, 0.9226],
        'Status': ['Weak', 'Fair', 'Strong', 'Best', 'Strong']
    }
    comp_df = pd.DataFrame(comparison_data)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    r2_chart = comp_df[['Model', 'R2']].set_index('Model')
    st.bar_chart(r2_chart)

    st.success("ANN achieved the highest R2 (0.933). Random Forest (0.926) is close and more stable.")
    st.info("Ref Paper 8: All our models decisively beat the Baseline (Mean Yield), proving ML effectiveness.")


# ═══════════ TAB 7: EXPLAINABLE AI (XAI) ═════════════════════════
with tab7:
    st.header("Explainable AI (XAI)")
    st.caption("Addressing the Black-Box problem by explaining which factors drive the yield predictions.")

    xai_df = get_feature_importance(model, encoder)
    if xai_df is not None:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("Global Feature Importance")
            # Top 10 features for cleaner UI
            chart_data = xai_df.head(10).set_index("Feature")[["Importance (%)"]]
            st.bar_chart(chart_data)
        with c2:
            st.subheader("Interpretation")
            st.info(generate_explanation_text(xai_df, top_n=4))
            st.write("*(Factors like Temperature, Rainfall, and Fertilizer typically show high impact, validating the model against agronomic principles.)*")
    else:
        st.warning("The current model does not support feature importance extraction.")


# ─── Footer ───────────────────────────────────────────────────────
st.divider()
st.caption("FarmTwin v2 -- Digital Twin-based AI for Agriculture | Conference Paper Prototype")
