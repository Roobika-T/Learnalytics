import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
import os

# Create directory if it doesn't exist
if not os.path.exists('Saved_models'):
    os.makedirs('Saved_models')

# Path to the Iris dataset
DATASET_PATH = 'Datasets/IRIS.csv'

# Preprocessing function
def preprocess_data():
    df = pd.read_csv(DATASET_PATH)
    X = df.drop("species", axis=1).values  # Assuming the target column is named 'species'
    y = df["species"].astype('category')
    y_codes = y.cat.codes.values
    species_mapping = dict(enumerate(y.cat.categories))  # Map codes to species names
    return X, y_codes, species_mapping

# Function to train SVM and save the model
def train_model():
    X, y, species_mapping = preprocess_data()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)  # Scale features

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create an SVM model
    model = SVC(kernel='rbf', C=1.0, gamma='scale')

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Display evaluation metrics
    print(f"Evaluation Metrics for Iris SVM Model:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Save the model, scaler, and species mapping
    model_filename = 'Saved_models/svm_model_iris.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump((model, scaler, species_mapping), f)

    return (f"Evaluation Metrics for Iris SVM Model:\nAccuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\n Recall: {recall:.4f}\nF1 Score: {f1:.4f}\nIris SVM model trained and saved.")

# Training process
if __name__ == "__main__":
    train_model()
