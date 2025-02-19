# gb_train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

def train_gradient_boosting_model(data_path, target_column, n_estimators, max_depth, learning_rate):
    # Load the dataset
    df = pd.read_csv(data_path)

    # Split the dataset into features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the Gradient Boosting Classifier
    model = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
    model.fit(X_train, y_train)

    # Save the model
    with open('gradient_boosting_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    class_report = classification_report(y_test, predictions)

    return X_test, y_test, accuracy, conf_matrix, class_report, "Model trained successfully!"