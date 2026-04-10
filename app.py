import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os

# Set page configuration
st.set_page_config(
    page_title="Calories Burnt Predictor",
    page_icon="🔥",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    .prediction-card {
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-val {
        font-size: 3rem;
        font-weight: bold;
        color: #ff4b4b;
    }
    </style>
    """, unsafe_allow_html=True)

# Load assets
@st.cache_resource
def load_assets():
    model = joblib.load('best_xgb_model.pkl')
    scaler = joblib.load('scaler.pkl')
    le = joblib.load('label_encoder.pkl')
    return model, scaler, le

try:
    model, scaler, le = load_assets()
except Exception as e:
    st.error(f"Error loading model assets: {e}")
    st.stop()

# Sidebar for inputs
st.sidebar.header("📋 User Input Features")

def user_input_features():
    gender = st.sidebar.selectbox("Gender", ("male", "female"))
    age = st.sidebar.slider("Age", 10, 100, 25)
    height = st.sidebar.slider("Height (cm)", 120, 230, 170)
    weight = st.sidebar.slider("Weight (kg)", 30, 150, 70)
    duration = st.sidebar.slider("Duration (min)", 1, 60, 20)
    heart_rate = st.sidebar.slider("Heart Rate (BPM)", 60, 200, 100)
    body_temp = st.sidebar.slider("Body Temperature (°C)", 36.0, 42.0, 39.0)
    
    # Feature Engineering
    bmi = weight / ((height / 100) ** 2)
    intensity_score = heart_rate * duration
    metabolic_stress = body_temp * heart_rate
    
    # Encode gender
    gender_encoded = 1 if gender == "female" else 0
    
    data = {
        'Gender': gender_encoded,
        'Age': age,
        'Height': float(height),
        'Weight': float(weight),
        'Duration': float(duration),
        'Heart_Rate': float(heart_rate),
        'Body_Temp': float(body_temp),
        'BMI': bmi,
        'Intensity_Score': float(intensity_score),
        'Metabolic_Stress': float(metabolic_stress)
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Main page layout
st.title("🔥 Calories Burnt Prediction App")
st.markdown("Predict how many calories you've burned based on your physical activity and metrics.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Your Input Metrics")
    st.write(input_df)
    
    if st.button("Predict Calories"):
        # Scale input
        input_scaled = scaler.transform(input_df)
        
        # Predict
        prediction = model.predict(input_scaled)
        
        st.markdown(f"""
            <div class="prediction-card">
                <h3>Estimated Calories Burned</h3>
                <div class="metric-val">{prediction[0]:.2f} kcal</div>
                <p>Keep up the great work!</p>
            </div>
        """, unsafe_allow_html=True)

with col2:
    st.subheader("📊 Visualizing Your Metrics")
    
    # Simple radar chart or bar chart for the user metrics
    metrics_data = {
        'Metric': ['Age', 'Height', 'Weight', 'Duration', 'Heart Rate'],
        'Value': [input_df['Age'][0], input_df['Height'][0], input_df['Weight'][0], 
                  input_df['Duration'][0], input_df['Heart_Rate'][0]]
    }
    viz_df = pd.DataFrame(metrics_data)
    
    fig = px.bar(viz_df, x='Metric', y='Value', color='Metric',
                 title="Input Metric Distribution",
                 template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# Bottom section with more graphs
st.divider()
st.subheader("📈 Insights from Training Data")

@st.cache_data
def load_sample_data():
    return pd.read_csv('calories.csv').sample(500)

try:
    df_sample = load_sample_data()
    
    row2_col1, row2_col2 = st.columns(2)
    
    with row2_col1:
        fig_dur = px.scatter(df_sample, x='Duration', y='Calories', color='Heart_Rate',
                            title="Calories vs Duration (Color by Heart Rate)",
                            template="plotly_white")
        st.plotly_chart(fig_dur, use_container_width=True)
        
    with row2_col2:
        fig_hr = px.scatter(df_sample, x='Heart_Rate', y='Calories', color='Body_Temp',
                           title="Calories vs Heart Rate (Color by Body Temp)",
                           template="plotly_white")
        st.plotly_chart(fig_hr, use_container_width=True)
except:
    st.info("Training data visualization unavailable. Ensure 'calories.csv' is in the project folder.")

st.sidebar.markdown("---")

