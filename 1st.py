from xml.parsers.expat import model
import kagglehub as kg
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.neural_network as mode
from sklearn.model_selection import train_test_split as trte
from sklearn.tree import plot_tree
# data 
data = pd.read_csv(r"C:\Users\dopha\.cache\kagglehub\datasets\neuralsorcerer\student-performance\versions\1\train.csv")

data1 =data.head(500)


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

def chat_showdata():

    
    print("start with the dataset?")
    c = input("")
    if c.lower()[0] == "y": 
        print("Here is the dataset:")
        print(data)
        nan = data[data.isna().any(axis=1)]
        print(nan)
        
        #sns.pairplot(data1) 
        #plt.show()
    
chat_showdata()

def train_model():
    
    model = mode.MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    
    print("Training the model...")

    model.fit( X_train, Y_train)
    
    print ("data is trained")
    print ("type data for: Age," \
    "Current grade" \
    "Gender (Male or Female)" \
    "Race (White / Hispanic / Black / Two-or-more / Asian)" \
    "SES_Quartile," \
    "ParentalEducation (HS/SomeCollege/Bachelors+/<HS)," \
    "SchoolType (public or private)," \
    "Locale (Suburban, City, Rural, Town)," \
    "TestScore_Math," \
    "TestScore_Reading," \
    "TestScore_Science," \
    "GPA,AttendanceRate," \
    "StudyHours," \
    "InternetAccess," \
    "Extracurricular," \
    "PartTimeJob," \
    "ParentSupport," \
    "Romantic," \
    "FreeTime," \
    "GoOut")

    v = model.predict(X_test)
    plt.figure(figsize=(20, 10))
    plot_tree(model.estimators_[0], feature_names=X_train.columns, filled=True)
    plt.show()
    print(v)

    
train_model()






