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
        
        # Add debug information about the model
        try:
            if hasattr(model, 'feature_names_in_'):
                print("Model expects features in this order:")
                for i, name in enumerate(model.feature_names_in_):
                    print(f"{i+1}. {name}")
        except Exception as e:
            print(f"Could not get feature names: {e}")
            
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Preprocess input data
def preprocess_input():
    # Map categorical variables to numerical values (must match training data)
    gender_map = {"Female": 0, "Male": 1}
    race_map = {
        "Black": 0, "White": 1, "Hispanic": 2, 
        "Two or more": 3, "Asian": 4, "Other": 5
    }
    edu_map = {
        "High School": 0, 
        "Less than High School": 1, 
        "Bachelor's Degree or Higher": 2, 
        "Some College": 3
    }
    school_map = {"Public": 0, "Private": 1}
    locale_map = {"Suburban": 0, "City": 1, "Town": 3, "Rural": 4}
    
    # Create input form
    st.header("Student Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        race = st.selectbox("Race/Ethnicity", 
                         ["Black", "White", "Hispanic", 
                          "Two or more", "Asian", "Other"])
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
    
    # Additional required fields with default values
    st.subheader("Additional Information")
    col3, col4 = st.columns(2)
    with col3:
        age = st.number_input("Age", min_value=10, max_value=30, value=18)
        internet_access = st.selectbox("Internet Access", ["Yes", "No"])
        parent_support = st.selectbox("Parental Support", ["Yes", "No"])
    with col4:
        part_time_job = st.selectbox("Part-time Job", ["Yes", "No"])
        romantic = st.selectbox("In a Relationship", ["Yes", "No"])
        
    # Create input dictionary with exact feature names from training
    # Note: The order of these features must match exactly with the training data
    input_data = {
        'Gender': gender_map[gender],
        'Race': race_map[race],
        'ParentalEducation': edu_map[parental_edu],
        'SchoolType': school_map[school_type],
        'Locale': locale_map[locale],
        'SES_Quartile': 2,  # Default middle quartile
        'StudyHours': 10,    # Default value
        'TestScore_Math': math_score,
        'TestScore_Reading': reading_score,
        'TestScore_Science': (math_score + reading_score) // 2,  # Estimate science score
        'AttendanceRate': attendance / 100,
        'Age': age,
        'InternetAccess': 1 if internet_access == "Yes" else 0,
        'ParentSupport': 1 if parent_support == "Yes" else 0,
        'PartTimeJob': 1 if part_time_job == "Yes" else 0,
        'Romantic': 1 if romantic == "Yes" else 0,
        'Grade': 10,  # Default value
        'Extracurricular': 0,
        'FreeTime': 0,
        'GoOut': 0
    }
    
    # Create a button for prediction
    submitted = st.button("Predict GPA")
    
    # Convert to DataFrame with columns in the correct order
    # Get the feature names in the order they were used during training
    # This is a best guess - you may need to adjust this order based on your training data
    feature_order = [
        'Gender', 'Race', 'ParentalEducation', 'SchoolType', 'Locale',
        'SES_Quartile', 'StudyHours', 'TestScore_Math', 'TestScore_Reading',
        'TestScore_Science', 'AttendanceRate', 'Age', 'InternetAccess',
        'ParentSupport', 'PartTimeJob', 'Romantic', 'Grade', 'Extracurricular',
        'FreeTime', 'GoOut'
    ]
    
    # Ensure all features are in the correct order
    ordered_data = {feature: input_data[feature] for feature in feature_order}
    
    return pd.DataFrame([ordered_data]), submitted

# Main app
def main():
    st.set_page_config(page_title="Student Performance Predictor", page_icon="")
    
    st.title("Student Performance Predictor")
    st.write("""
    This app predicts a student's GPA based on various factors.
    Please fill in the student's information below.
    """)
    
    # Load model
    model = load_model()
    
    if model is None:
        st.warning("Model not found. Please train the model first using 1st.py")
        return
    
    # Get input data from the form
    input_df, submitted = preprocess_input()
    
    if submitted:
        try:
            # Make prediction
            prediction = model.predict(input_df)
            
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
            
            attendance = input_df['AttendanceRate'].values[0] * 100  # Convert back to percentage
            if attendance < 95:
                st.write(f"→ Increase attendance rate (current: {attendance:.0f}%)")
                
            min_score = min(input_df['TestScore_Math'].values[0], 
                          input_df['TestScore_Reading'].values[0],
                          input_df['TestScore_Science'].values[0])
            if min_score < 70:
                st.write("→ Focus on improving the lowest test score")
                
            if input_df['StudyHours'].values[0] < 15:
                st.write(f"→ Increase study hours (current: {input_df['StudyHours'].values[0]} hours/week)")
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            st.warning("Please check if all required features are provided correctly.")

if __name__ == "__main__":
    main()
