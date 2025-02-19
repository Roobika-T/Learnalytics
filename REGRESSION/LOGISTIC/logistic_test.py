import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Function to test the logistic regression model
def test_logistic_regression_model(data_path, target_column):
    try:
        # Load the saved model
        with open('Saved_models/logistic_regression_model.pkl', 'rb') as model_file:
            regression_model = pickle.load(model_file)

        # Load dataset
        data = pd.read_csv(data_path)

        # Check if target column exists in the dataset
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in the dataset.")

        # Define features and target variable
        X = data.drop(target_column, axis=1)
        y = data[target_column]

        # Split data into 80% train and 20% test
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

        # Make predictions
        predictions = regression_model.predict(test_x)

        # Calculate evaluation metrics
        accuracy = accuracy_score(test_y, predictions)
        conf_matrix = confusion_matrix(test_y, predictions)
        class_report = classification_report(test_y, predictions)

        return accuracy, conf_matrix, class_report

    except Exception as e:
        print(f"An error occurred: {e}")