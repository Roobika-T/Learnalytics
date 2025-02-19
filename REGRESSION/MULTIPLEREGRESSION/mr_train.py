import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def preprocess_data(data, target_column):
    # Check for missing values
    missing_values = data.isnull().sum()
    print("Missing values:\n", missing_values)
    data = data.dropna()

    # Encode categorical variables
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != target_column:
            label_encoder = LabelEncoder()
            data[col] = label_encoder.fit_transform(data[col])

    # Check for outliers using IQR
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    outlier_condition = (data[numeric_cols] < (Q1 - 1.5 * IQR)) | (data[numeric_cols] > (Q3 + 1.5 * IQR))
    outliers = data[outlier_condition].copy()  # Store outliers for inspection
    print("Outliers detected:\n", outliers)
    # Remove outliers from the data
    data = data[~outlier_condition.any(axis=1)] 
    return data

def train_multiple_regression_model(data_path, target_column):
    # Load dataset
    data = pd.read_csv(data_path)
    data = preprocess_data(data, target_column)
    
    # Define features and target variable
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Split data into 80% train and 20% test
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Multiple Regression model
    model = LinearRegression()
    model.fit(train_x, train_y)

    # Save the model and the label encoder
    with open('Saved_models/multiple_regression_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)

    # Save the label encoder for the categorical variable
    label_encoder = LabelEncoder()
    label_encoder.fit(data['Extracurricular Activities'])  # Fit on the training data
    with open('Saved_models/label_encoder.pkl', 'wb') as le_file:
        pickle.dump(label_encoder, le_file)

    # Make predictions on the test set
    predictions = model.predict(test_x)

    # Calculate evaluation metrics
    mae = mean_absolute_error(test_y, predictions)
    mse = mean_squared_error(test_y, predictions)
    r2 = r2_score(test_y, predictions)

    return test_x, test_y, model, mse, mae, r2, "Model trained successfully!"