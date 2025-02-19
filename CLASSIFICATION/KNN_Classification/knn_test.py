import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Function to load the KNN model from the pickle file
def load_model():
    with open('Saved_models/knn_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model

# Function to classify a sample input using the loaded KNN model
def classify_iris(sepal_length, sepal_width, petal_length, petal_width):
    model = load_model()
    
    # Create a DataFrame for the input features
    input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], 
                               columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    
    # Optional: Scale the input data if your model was trained on scaled data
    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data)

    # Make a prediction
    prediction = model.predict(input_data_scaled)
    
    return prediction[0]

# Streamlit UI for user input
st.title("Iris Flower Classification using KNN")
st.write("Enter the features of the iris flower:")

sepal_length = st.number_input("Sepal Length", min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input("Sepal Width", min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.number_input("Petal Length", min_value=0.0, max_value=10.0, value=1.5)
petal_width = st.number_input("Petal Width", min_value=0.0, max_value=10.0, value=0.2)

if st.button("Classify"):
    result = classify_iris(sepal_length, sepal_width, petal_length, petal_width)
    st.success(f'The predicted species is: {result}')