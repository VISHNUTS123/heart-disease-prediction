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

# Feature and target separation
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Standardize the feature data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, stratify=Y, random_state=2)

# Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# Model accuracy
training_data_accuracy = accuracy_score(model.predict(X_train), Y_train)
test_data_accuracy = accuracy_score(model.predict(X_test), Y_test)

# -------------- Streamlit App --------------
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title('Heart Disease Prediction App')

# Heart image
img = Image.open('heart_img.jpg')
st.image(img, caption="Heart Health", use_container_width=False, width=200)

# Input field
st.subheader("Enter the 13 Feature Values")
input_text = st.text_input('Enter comma-separated values (13 total):')

# Placeholders for output
result_placeholder = st.empty()
input_summary_placeholder = st.empty()

# Show feature names and descriptions below the output
st.subheader("Feature Descriptions")
feature_info = {
    "1. Age": "Age in years",
    "2. Sex": "1 = Male, 0 = Female",
    "3. Chest Pain Type (cp)": "0 = Typical Angina, 1 = Atypical Angina, 2 = Non-anginal Pain, 3 = Asymptomatic",
    "4. Resting Blood Pressure (trestbps)": "In mm Hg",
    "5. Serum Cholesterol (chol)": "In mg/dl",
    "6. Fasting Blood Sugar (fbs)": "1 = > 120 mg/dl, 0 = ≤ 120 mg/dl",
    "7. Resting ECG results (restecg)": "0 = Normal, 1 = ST-T wave abnormality, 2 = Left ventricular hypertrophy",
    "8. Max Heart Rate Achieved (thalach)": "In bpm",
    "9. Exercise-induced Angina (exang)": "1 = Yes, 0 = No",
    "10. ST Depression (oldpeak)": "Induced by exercise relative to rest",
    "11. Slope of Peak Exercise ST Segment (slope)": "0 = Upsloping, 1 = Flat, 2 = Downsloping",
    "12. Number of Major Vessels (ca)": "0–3 colored by fluoroscopy",
    "13. Thalassemia (thal)": "1 = Normal, 2 = Fixed defect, 3 = Reversible defect"
}
for name, desc in feature_info.items():
    st.write(f"**{name}**: {desc}")

if input_text:
    try:
        input_list = [float(x) for x in input_text.strip().split(',')]
        if len(input_list) != X.shape[1]:
            result_placeholder.warning(f"Expected {X.shape[1]} values, but got {len(input_list)}.")
            input_summary_placeholder.empty()
        else:
            # Create descriptive summary for each feature
            feature_descriptions = {
                "Age": f"{input_list[0]} years",
                "Sex": "Male" if input_list[1] == 1 else "Female",
                "Chest Pain Type (cp)": {
                    0: "Typical Angina",
                    1: "Atypical Angina",
                    2: "Non-anginal Pain",
                    3: "Asymptomatic"
                }.get(int(input_list[2]), "Unknown"),
                "Resting Blood Pressure (trestbps)": f"{input_list[3]} mm Hg",
                "Serum Cholesterol (chol)": f"{input_list[4]} mg/dl",
                "Fasting Blood Sugar (fbs)": "> 120 mg/dl" if input_list[5] == 1 else "≤ 120 mg/dl",
                "Resting ECG results (restecg)": {
                    0: "Normal",
                    1: "ST-T wave abnormality",
                    2: "Left ventricular hypertrophy"
                }.get(int(input_list[6]), "Unknown"),
                "Max Heart Rate Achieved (thalach)": f"{input_list[7]} bpm",
                "Exercise-induced Angina (exang)": "Yes" if input_list[8] == 1 else "No",
                "ST Depression (oldpeak)": f"{input_list[9]} (exercise-induced ST depression)",
                "Slope of Peak Exercise ST Segment (slope)": {
                    0: "Upsloping",
                    1: "Flat",
                    2: "Downsloping"
                }.get(int(input_list[10]), "Unknown"),
                "Number of Major Vessels (ca)": f"{int(input_list[11])} (colored by fluoroscopy)",
                "Thalassemia (thal)": {
                    1: "Normal",
                    2: "Fixed defect",
                    3: "Reversible defect"
                }.get(int(input_list[12]), "Unknown"),
            }

            # Build markdown summary
            input_summary_md = ""
            for feature, desc in feature_descriptions.items():
                input_summary_md += f"**{feature}**: {desc}  \n"

            # Display summary
            input_summary_placeholder.markdown("### Input Summary")
            input_summary_placeholder.markdown(input_summary_md)

            # Predict and show result
            input_scaled = scaler.transform([input_list])
            prediction = model.predict(input_scaled)

            if prediction[0] == 0:
                result_placeholder.success("This person is unlikely to have heart disease.")
            else:
                result_placeholder.error("This person is likely to have heart disease.")
    except ValueError:
        result_placeholder.error("Please enter only numeric comma-separated values.")
        input_summary_placeholder.empty()

# Dataset preview and model metrics
st.subheader("Dataset Preview")
st.dataframe(heart_data.head())

st.subheader("Model Performance")
st.write(f"Training Accuracy: {training_data_accuracy:.2f}")
st.write(f"Test Accuracy: {test_data_accuracy:.2f}")

conf_matrix = confusion_matrix(Y_test, model.predict(X_test))
st.write("Confusion Matrix (Test Data):")
st.dataframe(pd.DataFrame(conf_matrix, columns=["Pred 0", "Pred 1"], index=["Actual 0", "Actual 1"]))
