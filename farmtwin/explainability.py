import pandas as pd
import numpy as np

def get_feature_importance(model, encoder):
    """
    Get global feature importance from a tree-based model (e.g., Random Forest).
    Returns a sorted DataFrame of features and their importance scores.
    """
    if not hasattr(model, 'feature_importances_'):
        return None
    
    # Numeric features (must match the order in simulation.py _build_input)
    num_features = ['Temperature_C', 'Rainfall_mm', 'Humidity_pct', 'Soil_Moisture_pct',
                  'Irrigation_mm', 'N_Fertilizer', 'P_Fertilizer', 'K_Fertilizer',
                  'Total_Water', 'Total_NPK', 'N_Ratio']
    
    # Categorical features
    cat_cols = ['Crop_Type', 'Soil_Type', 'Season', 'Location']
    cat_features = encoder.get_feature_names_out(cat_cols)
    
    # The final dataframe in _build_input is pd.concat([num_df, encoded_df], axis=1)
    all_features = num_features + list(cat_features)
    
    importances = model.feature_importances_
    
    if len(all_features) != len(importances):
        # Fallback if there's a mismatch
        all_features = [f"Feature_{i}" for i in range(len(importances))]
        
    df = pd.DataFrame({
        'Feature': all_features,
        'Importance': importances
    })
    
    # Calculate percentage
    df['Importance (%)'] = (df['Importance'] / df['Importance'].sum()) * 100
    df = df.sort_values('Importance', ascending=False).reset_index(drop=True)
    return df

def generate_explanation_text(feature_df, top_n=3):
    """
    Generate a human-readable explanation based on top features.
    """
    if feature_df is None or feature_df.empty:
        return "Model does not support feature importance explanation."
        
    top_features = feature_df.head(top_n)
    explanation = "Based on the Explainable AI (XAI) analysis, the model's prediction is primarily driven by: \n\n"
    
    for i, row in top_features.iterrows():
        feat = row['Feature'].replace('_', ' ')
        pct = row['Importance (%)']
        explanation += f"- **{feat}** ({pct:.1f}% impact)\n"
        
    explanation += "\nThis transparency helps build trust and allows farmers to focus on managing the most critical factors for yield improvement."
    return explanation
