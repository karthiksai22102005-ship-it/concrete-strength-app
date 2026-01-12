import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# --- 1. CONFIGURATION ---
FILE_NAME = "Concrete_Data (1).xls"

print("========================================")
print("   TRAINING HYBRID AI MODEL (XGB + MLP) ")
print("========================================")

# --- 2. LOAD DATA ---
print(f"[1/5] Loading {FILE_NAME}...")
if not os.path.exists(FILE_NAME):
    print(f"ERROR: {FILE_NAME} not found. Please verify the file name.")
    exit()

# Use xlrd engine for .xls files
data = pd.read_excel(FILE_NAME, engine='xlrd')

X = data.iloc[:, 0:8].values
y = data.iloc[:, 8].values

# Log Transform (Crucial part of your algorithm)
y = np.log1p(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=42
)

# --- 3. PREPROCESSING ---
print("[2/5] Scaling Data...")
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# --- 4. TRAIN XGBOOST ---
print("[3/5] Training XGBoost (Your Custom Parameters)...")
xgb = XGBRegressor(
    n_estimators=400,
    learning_rate=0.025,
    max_depth=4,
    subsample=0.75,
    colsample_bytree=0.75,
    min_child_weight=8,
    reg_alpha=0.3,
    reg_lambda=2.5,
    random_state=42,
    n_jobs=-1
)
xgb.fit(X_train, y_train)

# --- 5. TRAIN MLP (NEURAL NETWORK) ---
print("[4/5] Training Neural Network...")
mlp = Sequential([
    Input(shape=(8,)),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dense(1)
])

mlp.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mse"
)

early_stop = EarlyStopping(
    patience=30,
    restore_best_weights=True
)

history = mlp.fit(
    X_train_s,
    y_train,
    validation_split=0.2,
    epochs=800,
    batch_size=32,
    callbacks=[early_stop],
    verbose=0
)

# --- 6. EVALUATE (METRICS) ---
print("[5/5] Calculating Performance Metrics...")

# Hybrid Prediction Logic (0.6 * XGB + 0.4 * MLP)
train_pred = np.expm1(
    0.6 * xgb.predict(X_train) +
    0.4 * mlp.predict(X_train_s).flatten()
)

test_pred = np.expm1(
    0.6 * xgb.predict(X_test) +
    0.4 * mlp.predict(X_test_s).flatten()
)

y_train_true = np.expm1(y_train)
y_test_true = np.expm1(y_test)

def metrics(y_true, y_pred, n):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    adj_r2 = 1 - (1 - r2) * (len(y_true) - 1) / (len(y_true) - n - 1)
    evs = explained_variance_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    nrmse = rmse / (y_true.max() - y_true.min())
    return rmse, mae, mse, r2, adj_r2, mape, nrmse, evs

# Calculate Metrics
tr = metrics(y_train_true, train_pred, 8)
te = metrics(y_test_true, test_pred, 8)

print("\n" + "="*40)
print(f"TRAINING R2 Score: {tr[3]*100:.2f}%")
print(f"TESTING R2 Score:  {te[3]*100:.2f}%")
print("="*40 + "\n")

# --- 7. SAVE MODELS ---
print("Saving models for the Web App...")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(xgb, "xgb_model.pkl")
mlp.save("mlp_model.keras")
print("SUCCESS: 'scaler.pkl', 'xgb_model.pkl', and 'mlp_model.keras' are ready.")