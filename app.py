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
        "Two-or-more": 3, "Asian": 4, "Other": 5
    }
    edu_map = {
        "HS": 0, 
        "<HS": 1, 
        "Bachelors+": 2, 
        "SomeCollege": 3
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
                          "Two-or-more", "Asian", "Other"])
        parental_edu = st.selectbox("Parental Education Level",
                                 ["HS", "<HS",
                                  "Bachelors+", "SomeCollege"])
        school_type = st.selectbox("School Type", ["Public", "Private"])
        
    with col2:
        locale = st.selectbox("Locale", 
                            ["Suburban", "City", "Town", "Rural"])
        lunch = st.selectbox("Lunch Type", 
                           ["Standard", "Free/Reduced"])
        test_prep = st.selectbox("Test Preparation", 
                               ["None", "Completed"])
        attendance = st.slider("Attendance Rate (%)", 0, 100, 90)
    
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
        
    # Create input dictionary with exact feature names and order from training
    input_data = {
        'Age': age,
        'Grade': 10,  # Default value
        'Gender': gender_map[gender],
        'Race': race_map[race],
        'SES_Quartile': 2,  # Default middle quartile
        'ParentalEducation': edu_map[parental_edu],
        'SchoolType': school_map[school_type],
        'Locale': locale_map[locale],
        'AttendanceRate': attendance / 100,
        'StudyHours': 10,    # Default value
        'InternetAccess': 1 if internet_access == "Yes" else 0,
        'Extracurricular': 0,  # Default value
        'PartTimeJob': 1 if part_time_job == "Yes" else 0,
        'ParentSupport': 1 if parent_support == "Yes" else 0,
        'Romantic': 1 if romantic == "Yes" else 0,
        'FreeTime': 0,  # Default value
        'GoOut': 0,     # Default value
        
    }
    
    # Create a button for prediction
    submitted = st.button("Predict the GPA")
    
    return input_data, submitted

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
    input_data, submitted = preprocess_input()
    
    if submitted:
        try:
            # Convert input data to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Make prediction
            prediction = model.predict(input_df)
            
            # Display the prediction
            st.success(f"Predicted GPA: {prediction[0]:.2f}")
            
            # Add some interpretation
            if prediction[0] >= 3.5:
                st.success("Excellent performance! This student is likely to excel academically.")
            elif prediction[0] >= 3.0:
                st.info("Good performance. This student is doing well academically.")
            elif prediction[0] >= 2.0:
                st.warning("Average performance. Some improvement areas exist.")
            else:
                st.error("Below average performance. Consider additional support.")
            
            # Show the input values for reference
            with st.expander("View Input Values"):
                st.json(input_data)
                
            # Provide improvement suggestions
            attendance = input_df['AttendanceRate'].values[0] * 100  # Convert back to percentage
            if attendance < 95:
                st.write(f"→ Increase attendance rate (current: {attendance:.0f}%)")
                
            if input_df['StudyHours'].values[0] < 15:
                st.write(f"→ Increase study hours (current: {input_df['StudyHours'].values[0]} hours/week)")
                
            st.write("→ Focus on maintaining good study habits and attendance")
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            st.warning("Please check if all required features are provided correctly.")

if __name__ == "__main__":
    main()
