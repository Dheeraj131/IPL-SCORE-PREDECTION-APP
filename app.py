import streamlit as st
import numpy as np
import pickle
import os
import tensorflow as tf
from tensorflow import keras

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🏏 IPL Score Predictor",
    page_icon="🏏",
    layout="centered",
)

# ─────────────────────────────────────────────
# Load model & artifacts (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model  = keras.models.load_model("model/ipl_score_model.keras")
    with open("model/le_bat.pkl",  "rb") as f: le_bat  = pickle.load(f)
    with open("model/le_bowl.pkl", "rb") as f: le_bowl = pickle.load(f)
    with open("model/scaler.pkl",  "rb") as f: scaler  = pickle.load(f)
    return model, le_bat, le_bowl, scaler

model, le_bat, le_bowl, scaler = load_artifacts()

bat_teams  = list(le_bat.classes_)
bowl_teams = list(le_bowl.classes_)

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.title("🏏 IPL Score Predictor")
st.markdown("Predict the **final innings score** from the current match state using a Deep Learning model.")
st.divider()

col1, col2 = st.columns(2)
with col1:
    bat_team = st.selectbox("🟦 Batting Team", bat_teams)
with col2:
    bowl_team = st.selectbox("🟥 Bowling Team", [t for t in bowl_teams if t != bat_team])

st.markdown("#### 📊 Current Match State")
col3, col4, col5 = st.columns(3)
with col3:
    runs    = st.number_input("Runs Scored",    min_value=0,   max_value=300, value=85,  step=1)
    wickets = st.number_input("Wickets Fallen", min_value=0,   max_value=10,  value=3,   step=1)
with col4:
    overs        = st.number_input("Overs Completed",      min_value=5.0, max_value=20.0, value=10.0, step=0.1, format="%.1f")
    runs_last_5  = st.number_input("Runs in Last 5 Overs", min_value=0,   max_value=150,  value=40,   step=1)
with col5:
    wickets_last_5 = st.number_input("Wickets in Last 5 Overs", min_value=0, max_value=5, value=1, step=1)

st.divider()

if st.button("🎯 Predict Final Score", use_container_width=True, type="primary"):
    try:
        bat_enc  = le_bat.transform([bat_team])[0]
        bowl_enc = le_bowl.transform([bowl_team])[0]

        input_data   = np.array([[bat_enc, bowl_enc, runs, wickets, overs, runs_last_5, wickets_last_5]])
        input_scaled = scaler.transform(input_data)
        prediction   = model.predict(input_scaled, verbose=0)[0][0]
        predicted_score = max(int(round(prediction)), runs)  # can't predict less than current runs

        st.success(f"### 🏆 Predicted Final Score: **{predicted_score} runs**")

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Predicted Total", f"{predicted_score}")
        col_b.metric("Runs Still Needed", f"{predicted_score - runs}")
        col_c.metric("Overs Remaining", f"{round(20.0 - overs, 1)}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.divider()
st.caption("Model: Feed-forward DNN | Loss: Huber | Built with TensorFlow & Streamlit")
