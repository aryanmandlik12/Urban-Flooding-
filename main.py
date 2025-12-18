# streamlit_urban_flood_predictor_fastest.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
import os, time
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Urban Flood Predictor", layout="centered")

FEATURES = [
    "MonsoonIntensity","TopographyDrainage","RiverManagement","Deforestation","Urbanization",
    "ClimateChange","DamsQuality","Siltation","AgriculturalPractices","Encroachments",
    "IneffectiveDisasterPreparedness","DrainageSystems","CoastalVulnerability","Landslides",
    "Watersheds","DeterioratingInfrastructure","PopulationScore","WetlandLoss","InadequatePlanning",
    "PoliticalFactors"
]
TARGET = "FloodProbability"

st.title("🌧️ Flood Predictor")

# ----------------------
# Load dataset
# ----------------------
try:
    df = pd.read_csv("flood.csv")
except Exception as e:
    st.error(f"Cannot read flood.csv: {e}")
    st.stop()

# basic check
missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing:
    st.error(f"Dataset missing columns: {missing}")
    st.stop()

# ----------------------
# Quick visualizations
# ----------------------
st.header("Dataset Visualizations")

# 1) Target distribution
st.subheader("Flood Probability Distribution")
fig1, ax1 = plt.subplots(figsize=(6,3))
ax1.hist(df[TARGET].dropna(), bins=25, alpha=0.9)
ax1.set_xlabel("FloodProbability")
ax1.set_ylabel("Count")
ax1.set_title("Distribution of FloodProbability")
st.pyplot(fig1)

# 2) Top correlations with target
st.subheader("Top feature correlations with FloodProbability")
corrs = df[FEATURES + [TARGET]].corr()[TARGET].drop(TARGET).sort_values(key=lambda x: np.abs(x), ascending=False)
top_n = min(12, len(corrs))
fig2, ax2 = plt.subplots(figsize=(8,4))
corrs.head(top_n).plot(kind="bar", ax=ax2)
ax2.set_ylabel("Correlation with FloodProbability")
ax2.set_xlabel("Feature")
ax2.set_title(f"Top {top_n} features correlated with FloodProbability")
st.pyplot(fig2)

# 3) Correlation heatmap (features)
st.subheader("Feature correlation heatmap")
fig3, ax3 = plt.subplots(figsize=(10,8))
sns.heatmap(df[FEATURES].corr(), ax=ax3, cmap="coolwarm", center=0, fmt=".2f", square=True, cbar_kws={"shrink": .7})
ax3.set_title("Feature-to-feature Pearson correlation")
st.pyplot(fig3)

# 4) Interactive single-feature plot (histogram + boxplot)
st.subheader("Inspect a feature")
feature_choice = st.selectbox("Choose feature to inspect", FEATURES, index=0)
fig4, (ax4a, ax4b) = plt.subplots(1,2, figsize=(9,3))
ax4a.hist(df[feature_choice].dropna(), bins=25)
ax4a.set_title(f"{feature_choice} — distribution")
ax4a.set_xlabel(feature_choice)
ax4b.boxplot(df[feature_choice].dropna(), vert=False)
ax4b.set_title(f"{feature_choice} — boxplot")
st.pyplot(fig4)

st.markdown("---")

# ----------------------
# Prepare data & tiny model (train once per session)
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

# Train (cached in session_state so it won't re-run every interaction)
if "model" not in st.session_state:
    start = time.time()
    X_train, X_test, y_train, y_test, scaler = prepare_data(df, FEATURES, TARGET)
    model = build_tiny_model(X_train.shape[1])
    # single epoch training for instant speed
    model.fit(X_train, y_train, epochs=1, batch_size=64, verbose=0)
    st.session_state["model"] = model
    st.session_state["scaler"] = scaler
    st.success(f"✅ Model trained in {time.time() - start:.2f} sec (1 epoch, tiny model)")

model = st.session_state["model"]
scaler = st.session_state["scaler"]

# ----------------------
# User Inputs (Form) - after visualizations
# ----------------------
st.header("Manual Prediction Input")
st.markdown("Fill the fields below and press **Predict** to obtain the flood probability.")

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

    # small visual: gauge-like bar
    st.markdown("### Risk gauge")
    gauge_fig, gauge_ax = plt.subplots(figsize=(6,1.2))
    gauge_ax.barh([0], [pred], color="tab:blue", height=0.6)
    gauge_ax.set_xlim(0,1)
    gauge_ax.set_yticks([])
    gauge_ax.set_xlabel("Probability (0 → 1)")
    gauge_ax.set_title("Flood probability (higher is worse)")
    # add threshold markers
    gauge_ax.axvline(0.33, color="green", linestyle="--")
    gauge_ax.axvline(0.66, color="orange", linestyle="--")
    st.pyplot(gauge_fig)

    if pred < 0.33:
        st.success("Low probability ✅ — routine monitoring.")
    elif pred < 0.66:
        st.warning("Moderate probability ⚠️ — prepare contingency.")
    else:
        st.error("High probability 🚨 — immediate action recommended.")

st.markdown("---")

