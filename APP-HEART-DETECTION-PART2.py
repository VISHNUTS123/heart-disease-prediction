import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import streamlit as st
from PIL import Image

# Load the dataset
heart_data = pd.read_csv('heart_disease_data.csv')

# Splitting the data
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, stratify=Y, random_state=2)

# Model training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# Evaluate the model
training_data_accuracy = accuracy_score(model.predict(X_train), Y_train)
test_data_accuracy = accuracy_score(model.predict(X_test), Y_test)

# ----------------- Streamlit UI -----------------

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title('💓 Heart Disease Prediction App')

# Image
img = Image.open('heart_img.jpg')
st.image(img, caption="Heart Health", use_container_width=False, width=200)

# User input
st.header("🔢 Enter Patient Features")
input_text = st.text_input('Enter comma-separated values (13 features):')

if input_text:
    try:
        input_list = [float(x) for x in input_text.strip().split(',')]
        if len(input_list) != X.shape[1]:
            st.warning(f"⚠️ Expected {X.shape[1]} values, but got {len(input_list)}.")
        else:
            # Standardize and reshape
            input_scaled = scaler.transform([input_list])
            prediction = model.predict(input_scaled)

            if prediction[0] == 0:
                st.success("✅ This person is unlikely to have heart disease.")
            else:
                st.error("⚠️ This person is likely to have heart disease.")
    except ValueError:
        st.error("❌ Please enter only numeric comma-separated values.")

# Display Data and Metrics
st.subheader("📊 Dataset Preview")
st.dataframe(heart_data.head())

st.subheader("✅ Model Evaluation")
st.write(f"**Training Accuracy:** {training_data_accuracy:.2f}")
st.write(f"**Test Accuracy:** {test_data_accuracy:.2f}")

conf_matrix = confusion_matrix(Y_test, model.predict(X_test))
st.write("**Confusion Matrix (Test Data):**")
st.dataframe(pd.DataFrame(conf_matrix, columns=["Pred 0", "Pred 1"], index=["Actual 0", "Actual 1"]))
