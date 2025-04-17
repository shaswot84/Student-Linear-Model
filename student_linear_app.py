import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

def load_model():
    with open("linear_model.pkl","rb") as file:
        model, scaler, le = pickle.load(file=file)
        return model,scaler,le
    
def preprocessing_data(data, scaler, le):
    data["Extracurricular Activities"] = le.transform([data["Extracurricular Activities"]])
    df = pd.DataFrame(data)
    df_transformed = scaler.transform(df)
    return df_transformed

def predict_data(data):
    model,scaler,le= load_model()
    processed_df= preprocessing_data(data,scaler,le)
    prediction  =model.predict(processed_df)
    return prediction

def main():
    st.title("Student Performance Linear Model")
    st.write("Enter your data to get a prediction for performance")

    hours_studied  = st.number_input("Hours studeied?", min_value=1, max_value=10, value=5)
    previous_score = st.number_input("Previous Scores?",value = 70, min_value=40)
    extracurricular_activities = st.selectbox("Extra Curricular Activity",["Yes","No"])
    sleep_hours = st.number_input("Sleep Hours?",min_value=3,max_value=10)
    paper_practiced  = st.number_input("Sample Question Papers Practiced",value=10)

    if st.button("Predict your score"):
        user_data = {
            "Hours Studied":hours_studied,
            "Previous Scores":previous_score,
            "Extracurricular Activities":extracurricular_activities,
            "Sleep Hours":sleep_hours,
            "Sample Question Papers Practiced":paper_practiced
        }
        prediction = predict_data(user_data)
        st.success(f"Your prediction result is: {prediction}")
if __name__ == "__main__":
    main()