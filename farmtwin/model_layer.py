"""
FarmTwin v2 — Model Layer
Implements multiple ML models, ensemble stacking, baseline comparison, and evaluation.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# ═══════════════════════════════════════════════════════════════════
# 1. INDIVIDUAL MODELS
# ═══════════════════════════════════════════════════════════════════

def train_random_forest(X_train, y_train, n_estimators=200, random_state=42):
    """Train a Random Forest Regressor."""
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    print(f" Random Forest trained ({n_estimators} trees)")
    return model


def train_linear_model(X_train, y_train):
    """Train a Linear Regression baseline."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    print(f" Linear Regression trained")
    return model


def train_ann(X_train, y_train, epochs=100, batch_size=32):
    """
    Train a simple Artificial Neural Network (Dense layers).
    Uses scikit-learn MLPRegressor to avoid heavy TensorFlow dependency.
    Architecture: 64 → 32 → Output (matching Paper 1 concept)
    """
    from sklearn.neural_network import MLPRegressor
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=epochs,
        batch_size=batch_size,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        verbose=False
    )
    model.fit(X_train, y_train)
    print(f" ANN trained (Dense 64 → 32, {epochs} epochs)")
    return model


# ═══════════════════════════════════════════════════════════════════
# 2. ENSEMBLE: STACKING (🔥 Paper 1 approach)
# ═══════════════════════════════════════════════════════════════════

def stacking_model(X_train, y_train, X_test, rf_model, ann_model):
    """
    Stacking ensemble: combine RF + ANN predictions using a meta-learner.
    final_prediction = meta_model(RF_output, ANN_output)
    """
    # Generate base-model predictions on training data (using cross-val style)
    from sklearn.model_selection import cross_val_predict

    rf_train_pred = cross_val_predict(
        RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        X_train, y_train, cv=3
    )
    ann_train_pred = cross_val_predict(
        __import__('sklearn.neural_network', fromlist=['MLPRegressor']).MLPRegressor(
            hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
            max_iter=100, random_state=42, early_stopping=True, verbose=False
        ),
        X_train, y_train, cv=3
    )

    # Stack as meta-features
    meta_X_train = np.column_stack([rf_train_pred, ann_train_pred])

    # Meta-learner
    meta_model = LinearRegression()
    meta_model.fit(meta_X_train, y_train)

    # Generate test predictions
    rf_test_pred = rf_model.predict(X_test)
    ann_test_pred = ann_model.predict(X_test)
    meta_X_test = np.column_stack([rf_test_pred, ann_test_pred])

    stacked_pred = meta_model.predict(meta_X_test)
    print(f" Stacking ensemble built (RF + ANN → Linear meta-learner)")
    return stacked_pred, meta_model


# ═══════════════════════════════════════════════════════════════════
# 3. BASELINE
# ═══════════════════════════════════════════════════════════════════

def baseline_mean(y_train, y_test):
    """
    Baseline: predict the mean of training yield for everything.
    Paper 8 says many ML models can't beat this — we must prove ours can.
    """
    mean_yield = y_train.mean()
    baseline_pred = np.full(len(y_test), mean_yield)
    print(f" Baseline (mean yield): {mean_yield:.2f} kg/ha")
    return baseline_pred


# ═══════════════════════════════════════════════════════════════════
# 4. EVALUATION
# ═══════════════════════════════════════════════════════════════════

def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_r2(y_true, y_pred):
    """Calculate R² Score."""
    return r2_score(y_true, y_pred)


def evaluate_model(name, y_true, y_pred):
    """Evaluate and print metrics for a model."""
    rmse = calculate_rmse(y_true, y_pred)
    r2 = calculate_r2(y_true, y_pred)
    print(f"   {name:25s} | RMSE: {rmse:10.2f} | R²: {r2:.4f}")
    return {'model': name, 'RMSE': round(rmse, 2), 'R2': round(r2, 4)}


def compare_all_models(y_test, predictions_dict):
    """
    Compare all models side by side.
    predictions_dict = {'Random Forest': y_pred_rf, 'Linear': y_pred_lr, ...}
    """
    print("\n" + "=" * 65)
    print("  MODEL COMPARISON (Time-based validation)")
    print("=" * 65)
    results = []
    for name, y_pred in predictions_dict.items():
        results.append(evaluate_model(name, y_test, y_pred))
    print("=" * 65)
    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════
# 5. SAVE / LOAD
# ═══════════════════════════════════════════════════════════════════

def save_model(model, name, directory='models'):
    """Save a model to disk."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f'{name}.pkl')
    joblib.dump(model, path)
    print(f" Saved: {path}")
    return path





if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from farmtwin.data_layer import prepare_data

    # Prepare data
    X_train, X_test, y_train, y_test, encoder, scaler = prepare_data()

    # Train all models
    rf = train_random_forest(X_train, y_train)
    lr = train_linear_model(X_train, y_train)
    ann = train_ann(X_train, y_train, epochs=200)

    # Predictions
    pred_rf = rf.predict(X_test)
    pred_lr = lr.predict(X_test)
    pred_ann = ann.predict(X_test)

    # Stacking
    pred_stack, meta = stacking_model(X_train, y_train, X_test, rf, ann)

    # Baseline
    pred_baseline = baseline_mean(y_train, y_test)

    # Compare
    results = compare_all_models(y_test, {
        'Baseline (Mean)': pred_baseline,
        'Linear Regression': pred_lr,
        'Random Forest': pred_rf,
        'Neural Network (ANN)': pred_ann,
        'Stacking (RF+ANN) ': pred_stack,
    })

    # Save models
    save_model(rf, 'random_forest')
    save_model(lr, 'linear_regression')
    save_model(ann, 'neural_network')
    save_model(meta, 'stacking_meta')
    save_model(encoder, 'encoder')
    save_model(scaler, 'scaler')

    print("\n✅ All models trained and saved!")
    print(results.to_string(index=False))
