import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

def train_model():
    # Load the data
    df = pd.read_csv("Datasets/Mall_Customers.csv")
    
    # Select features and target
    X = df[['Age', 'Annual Income (k$)', 'Gender']]
    y = df['Spending Score (1-100)']
    
    # Map 'Gender' to numerical values
    gender_mapping = {'Male': 0, 'Female': 1}
    X['Gender'] = X['Gender'].map(gender_mapping)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Save the model
    with open('Saved_models/linear_regression_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    
    print("Model trained and saved successfully.")

train_model()