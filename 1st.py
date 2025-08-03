from xml.parsers.expat import model
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.ensemble as mode
from sklearn.model_selection import train_test_split as trte
from sklearn.metrics import r2_score, mean_absolute_error

from tqdm import tqdm
from sklearn.neural_network import MLPRegressor
import optuna as optu
# data 
data = pd.read_csv(r"C:\Users\Do Pham Tuan\.cache\kagglehub\datasets\neuralsorcerer\student-performance\versions\1\train.csv")

data = data.dropna()# Remove rows with missing values

# Gender
gender = {
    "Female" :0,
    "Male" :1,

}
data["Gender"] = data["Gender"].map(gender)

#Race
race = {
    "Black": 0,
    "White": 1,
    "Hispanic": 2,
    "Two-or-more": 3,
    "Asian": 4,
    "Other": 5,}
data ["Race"] = data["Race"].map(race)

#SES_Quartile
#Parental education
edu= { 
    "HS": 0,
    "<HS": 1,
    "Bachelors+": 2,
    "SomeCollege": 3,
}
data["ParentalEducation"]= data["ParentalEducation"].map(edu)

#schooltype
schooltype = {
    "Public": 0,
    "Private": 1,
}
data["SchoolType"] = data["SchoolType"].map(schooltype)
#place living

place = {
    "Suburban": 0,
    "City": 1,
    "Town": 3,
    "Rural": 4,
}
data["Locale"] = data["Locale"].map(place)

X = data.drop("GPA", axis=1) 

Y = data["GPA"]

X_train,X_test, Y_train, Y_test = trte(X, Y, test_size=0.2, random_state=42)

def check_data():
    nan = data[data.isna().any(axis=1)]
    print("row with not a value:", nan)    
check_data()
 
def train_model():
    try:
        model = MLPRegressor(hidden_layer_sizes=(23, 12, 53, 34),
                           max_iter=70,
                           random_state=65)
        
        print("Training the neural network model...please wait")
        model.fit(X_train, Y_train)
        
        # Add feature names to the model for reference
        model.feature_names_in_ = X_train.columns.tolist()
        
        # Save the trained model
        import joblib
        joblib.dump(model, 'student_performance_model.pkl')
        print("Model saved as 'student_performance_model.pkl'")
        
        # Print the feature names in order for debugging
        print("\nFeatures used in training (in order):")
        for i, col in enumerate(X_train.columns):
            print(f"{i+1}. {col}")
        
        print("Data is trained")
        v = model.predict(X_test)
        r2 = r2_score(Y_test, v)
        mae = mean_absolute_error(Y_test, v)
        
        print(f"R² score: {r2:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        
        # Create and display metrics table
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.axis('off')
        table_data = [
            ["Metric", "Value"],
            ["R² Score", f"{r2:.4f}"],
            ["Mean Absolute Error", f"{mae:.4f}"]
        ]
        table = ax.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.5, 1.5)
        plt.title("Model Evaluation Metrics", pad=20)
        plt.tight_layout()
        plt.show()
        
        return model
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

train_model()










