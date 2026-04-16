import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
import joblib

print("Loading dataset...")
df = pd.read_csv('FarmTwin_Yield_Dataset.csv')

# Features and target
X = df[['Crop_Type', 'Temperature_C', 'Rainfall_mm', 'Irrigation_mm', 'N_Fertilizer', 'P_Fertilizer', 'K_Fertilizer']]
y = df['Yield_kg_per_ha']

# One-hot encode Crop_Type
print("Preprocessing data...")
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_crops = encoder.fit_transform(X[['Crop_Type']])
crop_columns = encoder.get_feature_names_out(['Crop_Type'])
encoded_df = pd.DataFrame(encoded_crops, columns=crop_columns, index=X.index)

X_numeric = X.drop('Crop_Type', axis=1)
X_final = pd.concat([X_numeric, encoded_df], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

print("Training RandomForest model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"✅ R2 Score (Accuracy): {r2_score(y_test, y_pred):.4f}")
print(f"✅ Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")

# Save the model and encoder for the simulator
joblib.dump(model, 'farmtwin_model.pkl')
joblib.dump(encoder, 'farmtwin_encoder.pkl')
print("Model (farmtwin_model.pkl) and Encoder (farmtwin_encoder.pkl) saved successfully!")
