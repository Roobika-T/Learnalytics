import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Function to preprocess the data
def preprocess_data(data, target_column):
    print("Preprocessing")
    # Check for missing values
    missing_values = data.isnull().sum()
    print("Missing values Dropping\n")
    data = data.dropna()

    # Encode categorical variables
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != target_column:
            label_encoder = LabelEncoder()
            data[col] = label_encoder.fit_transform(data[col])

    # Check for outliers using IQR
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    Q1 = data[numeric_cols].quantile(0.25)
    Q3 = data[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    outlier_condition = (data[numeric_cols] < (Q1 - 1.5 * IQR)) | (data[numeric_cols] > (Q3 + 1.5 * IQR))

    # Remove outliers from the data
    data = data[~outlier_condition.any(axis=1)]

    return data

# Function to train the logistic regression model
def train_logistic_regression_model(data_path, target_column):
    print("Training logistic regression model")
    # Load dataset
    data = pd.read_csv(data_path)
    data = preprocess_data(data, target_column)

    

    # Define features and target variable
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    
    # Split data into 80% train and 20% test
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Logistic Regression model
    model = LogisticRegression(max_iter=1000)  # Increased max_iter for convergence
    model.fit(train_x, train_y)

    # Save the model
    with open('Saved_models/logistic_regression_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)

    print("Logistic Regression model trained and saved successfully.")

    # Evaluate the model
    predictions = model.predict(test_x)
    accuracy = accuracy_score(test_y, predictions)
    conf_matrix = confusion_matrix(test_y, predictions)
    class_report = classification_report(test_y, predictions)

    return accuracy, conf_matrix, class_report, "Model trained Successfully"