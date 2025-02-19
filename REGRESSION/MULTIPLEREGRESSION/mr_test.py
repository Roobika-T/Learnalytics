import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def test_multiple_regression_model(test_x, test_y):
    # Load the saved model
    with open('Saved_models/multiple_regression_model.pkl', 'rb') as model_file:
        regression_model = pickle.load(model_file)
    
    # Make predictions
    predictions = regression_model.predict(test_x)

    # Calculate evaluation metrics
    mae = mean_absolute_error(test_y, predictions)
    mse = mean_squared_error(test_y, predictions)
    r2 = r2_score(test_y, predictions)

    return mae, mse, r2, predictions

