# streamlit_urban_flood_predictor_fastest.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
import os, time

st.set_page_config(page_title="Urban Flood Predictor", layout="centered")

FEATURES = [
    "MonsoonIntensity","TopographyDrainage","RiverManagement","Deforestation","Urbanization",
    "ClimateChange","DamsQuality","Siltation","AgriculturalPractices","Encroachments",
    "IneffectiveDisasterPreparedness","DrainageSystems","CoastalVulnerability","Landslides",
    "Watersheds","DeterioratingInfrastructure","PopulationScore","WetlandLoss","InadequatePlanning",
    "PoliticalFactors"
]
TARGET = "FloodProbability"

st.title("🌧️ Flood Predictor Using Deep Learning")

# ----------------------
# Load dataset
# ----------------------
try:
    df = pd.read_csv("flood.csv")
except Exception as e:
    st.error(f"Cannot read flood.csv: {e}")
    st.stop()

# ----------------------
# Prepare data
# ----------------------
def prepare_data(df, features, target, test_size=0.2, seed=42):
    df2 = df[features + [target]].dropna().copy()
    X = df2[features].astype(float).values
    y = df2[target].astype(float).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler

def build_tiny_model(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(8, activation="relu"),
        layers.Dense(4, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

# ----------------------
# Train once, then reuse
# ----------------------
if "model" not in st.session_state:
    start = time.time()
    X_train, X_test, y_train, y_test, scaler = prepare_data(df, FEATURES, TARGET)
    model = build_tiny_model(X_train.shape[1])
    model.fit(X_train, y_train, epochs=1, batch_size=64, verbose=0)  # only 1 epoch!
    st.session_state["model"] = model
    st.session_state["scaler"] = scaler
    st.success(f"✅ Model trained in {time.time()-start:.2f} sec")

model = st.session_state["model"]
scaler = st.session_state["scaler"]

# ----------------------
# User Inputs (Form)
# ----------------------
st.header("Manual Prediction Input")
with st.form("predict_form"):
    cols = st.columns(2)
    user_vals = {}
    for i, feat in enumerate(FEATURES):
        col = cols[i % 2]
        default_val = float(np.nanmean(df[feat]))
        user_vals[feat] = col.number_input(feat, value=default_val, step=0.1, format="%.2f")
    submit = st.form_submit_button("Predict")

if submit:
    x = np.array([user_vals[f] for f in FEATURES]).reshape(1, -1)
    x_scaled = scaler.transform(x)
    pred = model.predict(x_scaled, verbose=0).ravel()[0]
    pred = float(np.clip(pred, 0.0, 1.0))

    st.metric("Predicted Flood Probability", f"{pred:.3f}")
    if pred < 0.33:
        st.success("Low probability ✅")
    elif pred < 0.66:
        st.warning("Moderate probability ⚠️")
    else:
        st.error("High probability 🚨")
