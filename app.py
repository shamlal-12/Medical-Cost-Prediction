import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load Models & Scaler
models_dir = "models"
try:
    scaler = pickle.load(open(os.path.join(models_dir, 'scaler.pkl'), 'rb'))
    poly_model = pickle.load(open(os.path.join(models_dir, 'poly_model.pkl'), 'rb'))
    poly_transform = pickle.load(open(os.path.join(models_dir, 'poly_transform.pkl'), 'rb'))
except FileNotFoundError as e:
    st.error(f"⚠️ Error: {e}. Ensure all model files exist in the `models` directory.")
    st.stop()

# Page Config
st.set_page_config(page_title="Medical Cost Predictor", layout="wide")

# Custom CSS for UI Enhancements
st.markdown("""
    <style>
        /* Page Background */
        body {
            background-color: #0e1117;
            color: white;
        }
        /* Title */
        .main-title {
            text-align: center;
            font-size: 40px;
            font-weight: bold;
            color: #F4A261;
        }
        /* Sidebar */
        .sidebar .sidebar-content {
            background-color: #1E1E2F;
            color: white;
        }
        /* Input Fields */
        input, select, textarea {
            border-radius: 10px !important;
        }
        /* Button Styling */
        .stButton>button {
            background: linear-gradient(45deg, #FF512F, #DD2476);
            color: white;
            border-radius: 10px;
            padding: 10px 20px;
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-title">💰 Medical Cost Prediction App</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar Navigation
st.sidebar.header("ML Lab Task")
page = st.sidebar.radio("Go to", ["📊 Upload & EDA",  "🩺 Predict Medical Costs"])

# Add user details
st.sidebar.markdown("### 👤 Sham Lal")
st.sidebar.markdown("🎓 **Degree:** BSAI")

if page == "📊 Upload & EDA":
    st.header("📊 Upload Dataset for Analysis")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Encode categorical values
        if "sex" in df.columns:
            df["sex"] = df["sex"].map({"male": 1, "female": 0})
        if "smoker" in df.columns:
            df["smoker"] = df["smoker"].map({"yes": 1, "no": 0})
        if "region" in df.columns:
            region_mapping = {"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3}
            df["region"] = df["region"].map(region_mapping)

        # Display Dataset
        st.write("### 📜 Data Preview")
        st.dataframe(df.head())

        # EDA Plots
        if not df.select_dtypes(include=[np.number]).empty:
            st.write("### 🔥 Heatmap")
            fig, ax = plt.subplots()
            sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

            st.write("### 📈 Scatter Plot")
            scatter_x = st.selectbox("Select X-axis", df.columns)
            scatter_y = st.selectbox("Select Y-axis", df.columns)
            fig, ax = plt.subplots()
            sns.scatterplot(x=df[scatter_x], y=df[scatter_y], ax=ax)
            st.pyplot(fig)

            st.write("### 📦 Box Plot")
            box_col = st.selectbox("Select Column for Box Plot", df.columns)
            fig, ax = plt.subplots()
            sns.boxplot(y=df[box_col], ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Dataset does not contain numerical columns for EDA.")

elif page == "🩺 Predict Medical Costs":
    st.header("🩺 Predict Your Medical Insurance Cost")

    # Input Fields with Icons
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("🔢 Age", min_value=18, max_value=100, value=30)
        bmi = st.number_input("⚖️ BMI", min_value=10.0, max_value=50.0, value=25.0)
        children = st.number_input("👶 Children", min_value=0, max_value=10, value=0)

    with col2:
        sex = st.selectbox("🚻 Sex", ["Male", "Female"])
        smoker = st.selectbox("🚬 Smoker", ["Yes", "No"])
        region = st.selectbox("📍 Region", ["Northeast", "Northwest", "Southeast", "Southwest"])

    # Encode categorical variables
    sex = 1 if sex == "Male" else 0
    smoker = 1 if smoker == "Yes" else 0
    region_mapping = {"Northeast": 0, "Northwest": 1, "Southeast": 2, "Southwest": 3}
    region = region_mapping[region]

    # Predict Button
    if st.button("💰 Predict Cost"):
        input_data = np.array([[age, sex, bmi, children, smoker, region]])
        try:
            input_scaled = scaler.transform(input_data)
            input_poly = poly_transform.transform(input_scaled)
            predicted_cost = poly_model.predict(input_poly)
            st.success(f'🎯 Estimated Medical Cost: **${predicted_cost[0]:,.2f}**')
        except Exception as e:
            st.error(f"Prediction Error: {e}")
