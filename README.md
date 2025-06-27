# Explainable-Machine-Learning-Approaches-for-Cardiovascular-Disease-Risk-Assessment


This project implements a **Cardiovascular Disease Prediction Model** using a **Random Forest Classifier**. The model is deployed in a **Streamlit web app**, allowing users to input their health metrics and receive a prediction on their cardiovascular risk. Additionally, we incorporate **Explainable AI (XAI)** using SHAP to provide insights into the model's decision-making process.

---

## Features

- **Prediction**: The app predicts the likelihood of cardiovascular disease based on user inputs such as age, blood pressure, BMI, etc.
- **Explainability**: SHAP values are used to visualize the impact of each feature on the model’s decision.
- **User-friendly Interface**: The app is built with Streamlit, providing a simple and interactive UI for users.

---

## Tech Stack

- **Machine Learning Model**: Random Forest Classifier (scikit-learn)
- **XAI Library**: SHAP for model explanation
- **Web App**: Streamlit for UI
- **Model Serialization**: Joblib for saving/loading the model
- **Python Libraries**: pandas, numpy, scikit-learn, shap, matplotlib, streamlit

---
## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/22sakshiagarwal/Explainable-Machine-Learning-Approaches-for-Cardiovascular-Disease-Risk-Assessment.git
   cd Explainable-Machine-Learning-Approaches-for-Cardiovascular-Disease-Risk-Assessment

---
## Install dependencies
 pip install -r requirements.txt
---
## Usage
 1.Run the Streamlit app

 streamlit run app.py
---
