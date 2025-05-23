# Heart Disease Prediction Web App

This is a simple **Heart Disease Prediction** web application built using Python, Streamlit, and a Logistic Regression machine learning model.  
It predicts the presence of heart disease based on user-provided health features.

---

## Features

- Predicts heart disease risk using thirteen health parameters.
- Interactive Streamlit web interface for easy input and results display.
- Shows model accuracy on training and testing datasets.
- Displays the original dataset used for training.
- Includes an image for a better UI experience.

---

## Technologies Used

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Pillow (PIL)

---

## Dataset

The app uses a heart disease dataset (`heart_disease_data.csv`) with 13 features such as:

- `age`
- `sex`
- `cp` (chest pain type)
- `trestbps` (resting blood pressure)
- `chol` (cholesterol)
- `fbs` (fasting blood sugar)
- `restecg` (resting electrocardiographic results)
- `thalach` (maximum heart rate achieved)
- `exang` (exercise induced angina)
- `oldpeak` (ST depression induced by exercise)
- `slope` (slope of the peak exercise ST segment)
- `ca` (number of major vessels colored by fluoroscopy)
- `thal` (thalassemia)

The target column (`target`) indicates the presence (1) or absence (0) of heart disease.

---

Usage
Enter 13 comma-separated feature values in the input box.

The app will predict whether the person has heart disease or not.

View the dataset and model performance metrics below.

heart-disease-prediction/
│
├── APP-HEART-DETECTION.py    # Main Streamlit app script
├── heart_disease_data.csv    # Dataset used for training and testing
├── heart_img.jpg             # Image displayed on the app
├── requirements.txt          # Python dependencies list
└── README.md                 # This file


