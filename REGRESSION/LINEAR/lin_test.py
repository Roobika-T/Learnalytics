import pandas as pd

def Predict(model, input_data):
    return model.predict(input_data)

if __name__ == "__main__":
    # This section is for testing the model independently
    import pickle
    
    # Load the model
    with open('Saved_models/linear_regression_model.pkl', 'rb') as file:
        model = pickle.load(file)
    
    # Example input data
    test_data = pd.DataFrame({
        'Age': [25, 40],
        'Annual Income (k$)': [60, 80],
        'Gender': ['Male', 'Female']
    })
    
    # Make predictions
    predictions = Predict(model, test_data)
    print("Test Predictions:", predictions)
