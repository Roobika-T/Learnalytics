# gb_test.py
import pickle
import pandas as pd

def test_gradient_boosting_model(input_data):
    # Load the trained model
    with open('gradient_boosting_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    # Make predictions
    predictions = model.predict(input_data)
    
    return predictions