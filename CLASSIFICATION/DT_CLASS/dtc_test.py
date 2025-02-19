# test.py
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from CLASSIFICATION.DT_CLASS.dtc_train import load_data, predict

# Load dataset and split it
X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the saved model
try:
    with open("Saved_models/decision_tree_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Model not found. Please ensure the model has been trained and saved.")
    exit()

# Make predictions on the test set
y_pred = predict(model, X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Display the results
print("Evaluation Metrics on Test Set")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
