import streamlit as st
import numpy as np
import tensorflow as tf 
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Page config
st.set_page_config(page_title="Churn Prediction", page_icon="📊", layout="centered")

# Load model and preprocessors
@st.cache_resource
def load_model_and_preprocessors():
    model = tf.keras.models.load_model('best_churn_model.h5')
    
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    with open('label_encoder_gender.pkl', 'rb') as file:
        label_encoder = pickle.load(file)
    with open('ohe_geo.pkl', 'rb') as file:
        onehot_encoder = pickle.load(file)
    
    return model, scaler, label_encoder, onehot_encoder

model, scaler, label_encoder, onehot_encoder = load_model_and_preprocessors()

# App title and description
st.title("🏦 Customer Churn Prediction")
st.markdown("Predict whether a customer is likely to leave the bank based on their profile.")

# Add threshold slider in sidebar
st.sidebar.header("⚙️ Model Settings")
threshold = st.sidebar.slider(
    "Churn Risk Threshold", 
    min_value=0.1, 
    max_value=0.9, 
    value=0.35,  # More aggressive default
    step=0.05,
    help="Lower = more aggressive (catch more churners, more false alarms). Higher = more conservative."
)
st.sidebar.info(f"Current threshold: {threshold:.0%}")
st.sidebar.caption("🎯 Default (0.35) is more aggressive than standard (0.50)")

st.markdown("---")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 Demographics")
    geography = st.selectbox("Geography", onehot_encoder.categories_[0])
    gender = st.selectbox("Gender", label_encoder.classes_)
    age = st.slider("Age", min_value=18, max_value=100, value=35)
    
    st.subheader("💰 Financial Info")
    credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=650, step=10)
    balance = st.number_input("Balance ($)", min_value=0.0, value=50000.0, step=1000.0, format="%.2f")
    estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, value=50000.0, step=1000.0, format="%.2f")

with col2:
    st.subheader("🏦 Banking Relationship")
    tenure = st.slider("Tenure (years)", min_value=0, max_value=10, value=5)
    num_of_products = st.slider("Number of Products", min_value=1, max_value=4, value=1)
    has_cr_card = st.selectbox("Has Credit Card", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    is_active_member = st.selectbox("Is Active Member", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

st.markdown("---")

# Predict button
if st.button("🔮 Predict Churn", type="primary", use_container_width=True):
    # Prepare input data
    input_data = pd.DataFrame({
        "CreditScore": [credit_score],
        "Geography": [geography],
        "Gender": [gender],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [has_cr_card],
        "IsActiveMember": [is_active_member],
        "EstimatedSalary": [estimated_salary]
    })

    # One-hot encode Geography
    geo_encoded = onehot_encoder.transform([[geography]])
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder.get_feature_names_out(["Geography"]))

    # Label encode Gender
    input_data['Gender'] = label_encoder.transform([gender])

    # Drop Geography and concat the OHE geography columns
    input_data = pd.concat([input_data.drop(columns=['Geography']), geo_encoded_df], axis=1)

    # Ensure correct column order to match training
    expected_columns = ['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
                        'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 
                        'Geography_France', 'Geography_Germany', 'Geography_Spain']

    input_data = input_data[expected_columns]

    # Scale the data
    input_data_scaled = scaler.transform(input_data)

    # Predict
    with st.spinner('Analyzing customer profile...'):
        prediction = model.predict(input_data_scaled, verbose=0)
        prediction_prob = prediction[0][0]

    # Display result with nice formatting
    st.markdown("### Prediction Result")
    
    # Use custom threshold
    if prediction_prob > threshold:
        st.error(f"⚠️ **High Churn Risk**")
        st.metric("Churn Probability", f"{prediction_prob*100:.1f}%", delta=f"+{(prediction_prob-threshold)*100:.1f}% above threshold")
        
        # Risk level indicator
        if prediction_prob > 0.7:
            risk_level = "🔴 CRITICAL"
            recommendation = "**Immediate Action Required:** Contact customer urgently with retention offer."
        elif prediction_prob > 0.5:
            risk_level = "🟠 HIGH"
            recommendation = "**Action Recommended:** Proactive outreach and personalized retention strategy."
        else:
            risk_level = "🟡 ELEVATED"
            recommendation = "**Monitor Closely:** Consider preventive engagement."
        
        st.warning(f"Risk Level: {risk_level}")
        st.info(recommendation)
        
    else:
        st.success(f"✅ **Low Churn Risk**")
        st.metric("Churn Probability", f"{prediction_prob*100:.1f}%", delta=f"-{(threshold-prediction_prob)*100:.1f}% below threshold", delta_color="inverse")
        
        if prediction_prob < 0.2:
            st.info("**Status:** Highly stable customer. Continue standard engagement.")
        else:
            st.info("**Status:** Stable but monitor. Standard retention practices apply.")
    
    # Show raw probability
    st.markdown("---")
    st.caption(f"📊 Raw Model Output: {prediction_prob:.4f} | Threshold: {threshold:.2f}")
    
    # Confidence indicator
    confidence = abs(prediction_prob - threshold) / threshold
    if confidence > 0.5:
        st.caption(f"🎯 Prediction Confidence: High")
    elif confidence > 0.2:
        st.caption(f"🎯 Prediction Confidence: Medium")
    else:
        st.caption(f"⚠️ Prediction Confidence: Low (near threshold)")

# Add quick test profiles
with st.expander("🧪 Quick Test Profiles"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔴 High Risk Profile**")
        st.caption("Credit: 500 | Balance: $120 | Tenure: 0 | Inactive")
    
    with col2:
        st.markdown("**🟢 Low Risk Profile**")
        st.caption("Credit: 800 | Balance: $150k | Tenure: 10 | Active")

# Footer
st.markdown("---")
st.caption(f"Built with Streamlit • Model: ANN (Recall: 67.6%, AUC: 0.85) • Threshold: {threshold:.0%}")