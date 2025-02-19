import numpy as np
import pickle

# Function to load the model for prediction
def load_model():
    model_filename = 'Saved_models/svm_model_iris.pkl'
    with open(model_filename, 'rb') as f:
        model, scaler, species_mapping = pickle.load(f)
    return model, scaler, species_mapping

# Function for prediction
def predict_model(input_data):
    model, scaler, species_mapping = load_model()
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    return species_mapping[prediction[0]]  # Convert numeric prediction to species name

if __name__ == "__main__":
    # Input features for prediction
    sepal_length = float(input("Enter Sepal Length: "))
    sepal_width = float(input("Enter Sepal Width: "))
    petal_length = float(input("Enter Petal Length: "))
    petal_width = float(input("Enter Petal Width: "))
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    prediction = predict_model(input_data)
    print(f"Prediction: {prediction}")
