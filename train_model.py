"""
train_model.py
──────────────
Run this script once to train the model and save all artifacts to model/.

Usage:
    python train_model.py

Requires: ipl_data.csv in the project root.
Outputs (inside model/):
    ipl_score_model.keras
    le_bat.pkl
    le_bowl.pkl
    scaler.pkl
"""

import os
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(f"TensorFlow Version : {tf.__version__}")
print(f"Keras Version      : {keras.__version__}")

# ── 1. Load data ────────────────────────────────────────────────────────────
df = pd.read_csv("ipl_data.csv")
print(f"\nDataset shape: {df.shape}")

# ── 2. Filter & select features ─────────────────────────────────────────────
df_filtered = df[df["overs"] >= 5.0].copy()

feature_cols = ["bat_team", "bowl_team", "runs", "wickets",
                "overs", "runs_last_5", "wickets_last_5"]
target_col   = "total"

df_model = df_filtered[feature_cols + [target_col]].dropna()
print(f"Rows after filtering: {len(df_model)}")

# ── 3. Encode teams ──────────────────────────────────────────────────────────
le_bat  = LabelEncoder()
le_bowl = LabelEncoder()

df_model["bat_team_enc"]  = le_bat.fit_transform(df_model["bat_team"])
df_model["bowl_team_enc"] = le_bowl.fit_transform(df_model["bowl_team"])

os.makedirs("model", exist_ok=True)
with open("model/le_bat.pkl",  "wb") as f: pickle.dump(le_bat,  f)
with open("model/le_bowl.pkl", "wb") as f: pickle.dump(le_bowl, f)

print(f"\nBatting teams  ({len(le_bat.classes_)}): {list(le_bat.classes_)}")
print(f"Bowling teams  ({len(le_bowl.classes_)}): {list(le_bowl.classes_)}")

# ── 4. Scale ─────────────────────────────────────────────────────────────────
numeric_features = ["bat_team_enc", "bowl_team_enc", "runs", "wickets",
                    "overs", "runs_last_5", "wickets_last_5"]

X = df_model[numeric_features].values
y = df_model[target_col].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

with open("model/scaler.pkl", "wb") as f: pickle.dump(scaler, f)
print(f"\nX_train: {X_train_scaled.shape}  |  X_test: {X_test_scaled.shape}")

# ── 5. Build model ───────────────────────────────────────────────────────────
def build_model(input_dim: int) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),

        layers.Dense(512, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Dense(128, activation="relu"),
        layers.Dropout(0.1),

        layers.Dense(64, activation="relu"),

        layers.Dense(1, activation="linear"),   # regression output
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=["mae"],
    )
    return model

model = build_model(X_train_scaled.shape[1])
model.summary()

# ── 6. Train ─────────────────────────────────────────────────────────────────
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
)

history = model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=64,
    validation_data=(X_test_scaled, y_test),
    callbacks=[early_stop, reduce_lr],
    verbose=1,
)

# ── 7. Evaluate ──────────────────────────────────────────────────────────────
y_pred = model.predict(X_test_scaled).flatten()
mae    = mean_absolute_error(y_test, y_pred)
rmse   = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n==============================")
print("       MODEL EVALUATION       ")
print("==============================")
print(f"  MAE  : {mae:.2f} runs")
print(f"  RMSE : {rmse:.2f} runs")
print("==============================")

# ── 8. Save model ────────────────────────────────────────────────────────────
model.save("model/ipl_score_model.keras")
print("\n✅ Model saved → model/ipl_score_model.keras")
print("\nSaved artifacts:")
for fname in sorted(os.listdir("model")):
    size = os.path.getsize(f"model/{fname}")
    print(f"  📁 model/{fname}  ({size:,} bytes)")
