import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Set Streamlit page config
st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="centered")

# Load ANN model, scaler, and encoders
@st.cache_resource
def load_ann_model():
    model = load_model("diabetes_ann_model.h5")
    scaler = joblib.load("scaler.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    return model, scaler, label_encoders

# Load dataset (for UI)
@st.cache_data
def load_data():
    return pd.read_csv("diabetes_prediction_dataset.csv")

model, scaler, label_encoders = load_ann_model()
df = load_data()

st.title("🧠 Diabetes Risk Predictor (ANN)")
st.markdown("Predict your likelihood of diabetes using an **Artificial Neural Network (ANN)**.")
st.markdown("---")

# -------------------------------
# Gender Filter
# -------------------------------
st.subheader("👤 Filter by Gender")
if 'gender' in df.columns:
    gender_options = df['gender'].dropna().unique().tolist()
    selected_gender = st.radio("Select Gender:", gender_options, horizontal=True)
    filtered_df = df[df['gender'] == selected_gender].copy()
else:
    st.error("❌ 'gender' column not found.")
    st.stop()

# -------------------------------
# Top Correlated Features
# -------------------------------
if 'diabetes' in filtered_df.columns:
    st.subheader("📊 Top 3 Correlated Features with Diabetes")
    correlations = filtered_df.corr(numeric_only=True)['diabetes'].drop('diabetes').sort_values(ascending=False)
    st.dataframe(correlations.head(3).to_frame(name="Correlation"))
else:
    st.error("❌ 'diabetes' column not found.")
    st.stop()

st.markdown("---")

# -------------------------------
# Health Input Form
# -------------------------------
st.subheader("🧪 Enter Your Health Details")

with st.form("prediction_form"):
    input_data = {}
    for col in df.drop("diabetes", axis=1).columns:
        if col == "gender":
            input_data[col] = selected_gender  # Use selected gender
            continue
        if df[col].dtype == 'object':
            options = df[col].dropna().unique().tolist()
            input_data[col] = st.selectbox(f"{col}:", options)
        else:
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            mean_val = float(df[col].mean())
            if min_val == max_val:
                input_data[col] = st.number_input(f"{col}:", value=mean_val)
            else:
                input_data[col] = st.slider(f"{col}:", min_value=min_val, max_value=max_val, value=mean_val)

    submitted = st.form_submit_button("🔍 Predict")

# -------------------------------
# Prediction
# -------------------------------
if submitted:
    input_df = pd.DataFrame([input_data])

    # Encode categorical variables
    for col in input_df.select_dtypes(include='object').columns:
        input_df[col] = label_encoders[col].transform(input_df[col])

    # Scale numerical features
    input_scaled = scaler.transform(input_df)

    # Model Prediction
    prediction = model.predict(input_scaled)[0][0]
    predicted_class = int(prediction >= 0.5)

    # Result
    st.success("✅ Prediction Complete")
    if predicted_class == 1:
        st.markdown("### 🟥 **High Risk of Diabetes**")
    else:
        st.markdown("### 🟩 **Low Risk of Diabetes**")

    st.markdown(f"**🔢 Probability Score:** `{prediction:.2%}`")
    st.markdown("---")
    st.caption("📌 *Prediction made using a pre-trained ANN model on health data.*")
