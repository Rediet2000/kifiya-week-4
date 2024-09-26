import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

# Step 1: Train and Save the Model
def train_and_save_model():
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Decision Tree Classifier
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # # Save the trained model to a .joblib file
    # joblib.dump(model, 'model.pkl')
    # print("Model saved as 'model.joblib'")

# Train and save the model if it hasn't been saved yet
train_and_save_model()

# Step 2: Load the trained model
model = joblib.load('../notebooks/model/model.joblib')
iris = load_iris()  # Load the Iris dataset to get class names

# Step 3: Create an instance of FastAPI
app = FastAPI()

# Define the input data model
class InputData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Define the health check endpoint
@app.get("/")
async def root():
    return {"message": " Use the /predict/ endpoint to obtain Iris classification predictions"}

#  Define the prediction endpoint (POST request)

@app.post('/predict/')
async def predict(input_data: InputData):
    data = pd.DataFrame([input_data.model_dump()])

    # Make prediction
    prediction = model.predict(data)

    # Return the prediction as a response
    return {
        "prediction": int(prediction[0]),
        "class_name": iris.target_names[prediction[0]].tolist()  
        }
