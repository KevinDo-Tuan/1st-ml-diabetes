import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Load the trained model
def load_model():
    # We'll modify the train_model function to save and load the model
    try:
        model = joblib.load('student_performance_model.pkl')
        return model
    except:
        return None

# Preprocess input data
def preprocess_input(gender, race, parental_edu, school_type, locale, 
                   lunch, test_prep, attendance, math_score, reading_score, writing_score):
    # Map categorical variables to numerical values
    gender_map = {"Female": 0, "Male": 1}
    race_map = {
        "Black": 0, "White": 1, "Hispanic": 2, 
        "Two or more races": 3, "Asian": 4, "Other": 5
    }
    edu_map = {"High School": 0, "Less than High School": 1, 
              "Bachelor's Degree or Higher": 2, "Some College": 3}
    school_map = {"Public": 0, "Private": 1}
    locale_map = {"Suburban": 0, "City": 1, "Town": 3, "Rural": 4}
    
    # Create input array
    input_data = {
        'Gender': gender_map[gender],
        'Race': race_map[race],
        'ParentalEducation': edu_map[parental_edu],
        'SchoolType': school_map[school_type],
        'Locale': locale_map[locale],
        'Lunch': 1 if lunch == "Free/Reduced" else 0,
        'TestPrep': 1 if test_prep == "Completed" else 0,
        'Attendance': attendance,
        'MathScore': math_score,
        'ReadingScore': reading_score,
        'WritingScore': writing_score
    }
    
    return pd.DataFrame([input_data])

# Main app
def main():
    st.set_page_config(page_title="Student Performance Predictor", page_icon="")
    
    st.title("Student Performance Predictor")
    st.write("""
    This app predicts a student's GPA based on various factors.
    Please fill in the student's information below.
    """)
    
    # Create input form
    with st.form("student_info"):
        st.header("Student Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            race = st.selectbox("Race/Ethnicity", 
                             ["Black", "White", "Hispanic", 
                              "Two or more races", "Asian", "Other"])
            parental_edu = st.selectbox("Parental Education Level",
                                     ["High School", "Less than High School",
                                      "Bachelor's Degree or Higher", "Some College"])
            school_type = st.selectbox("School Type", ["Public", "Private"])
            
        with col2:
            locale = st.selectbox("Locale", 
                                ["Suburban", "City", "Town", "Rural"])
            lunch = st.selectbox("Lunch Type", 
                               ["Standard", "Free/Reduced"])
            test_prep = st.selectbox("Test Preparation", 
                                   ["None", "Completed"])
            attendance = st.slider("Attendance Rate (%)", 0, 100, 90)
        
        st.subheader("Test Scores (0-100)")
        score_col1, score_col2, score_col3 = st.columns(3)
        with score_col1:
            math_score = st.number_input("Math Score", 0, 100, 70)
        with score_col2:
            reading_score = st.number_input("Reading Score", 0, 100, 75)
        with score_col3:
            writing_score = st.number_input("Writing Score", 0, 100, 72)
        
        submitted = st.form_submit_button("Predict GPA")
    
    # Load model
    model = load_model()
    
    if submitted:
        if model is None:
            st.warning("Model not found. Please train the model first using 1st.py")
        else:
            # Preprocess input
            input_data = preprocess_input(
                gender, race, parental_edu, school_type, locale,
                lunch, test_prep, attendance, 
                math_score, reading_score, writing_score
            )
            
            # Make prediction
            prediction = model.predict(input_data)
            
            # Display results
            st.subheader("Prediction Results")
            st.metric(label="Predicted GPA", value=f"{prediction[0]:.2f}")
            
            # Add some interpretation
            if prediction[0] >= 3.5:
                st.success("Excellent performance! This student is likely to excel academically.")
            elif prediction[0] >= 2.5:
                st.info("Good performance. This student is on the right track!")
            else:
                st.warning("May need additional support. Consider academic interventions.")
            
            # Show feature importance (if available)
            st.subheader("Tips for Improvement")
            st.write("To improve the student's predicted GPA:")
            if attendance < 95:
                st.write("→ Increase attendance rate (current: {}%)".format(attendance))
            if min(math_score, reading_score, writing_score) < 70:
                st.write("→ Focus on improving the lowest test score")
            if test_prep == "None":
                st.write("→ Consider completing test preparation")

if __name__ == "__main__":
    main()
