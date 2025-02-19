import pickle
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def preprocess_data(data, target_column):
    # Check for missing values
    missing_values = data.isnull().sum()
    # print("Missing values:\n", missing_values)
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
    # print("Outliers detected:\n", outliers)
    # Remove outliers from the data
    data = data[~outlier_condition.any(axis=1)] 
    return data


def meta_data(data_choice):

    if data_choice == 1:
        feature_column=['suburb','postalCode','propType','bed', 'bath', 'car']
        target_column='sellPrice'
        return feature_column, target_column

def handle_unknown(data_choice, modes, user_input):
    
    if data_choice==1:  #SydneyHouseprices
        for column in user_input:
            if user_input[column] is None:
                # Access the mode directly from the dictionary using the column name
                if column in modes.keys():
                    user_input[column] = modes[column]
    return user_input

def train_decision_tree_model(data_path, target_column, max_depth, min_samples_split, min_samples_leaf):
    # Load dataset
    data = pd.read_csv(data_path)
    data = preprocess_data(data, target_column)
    X = data.drop(target_column, axis=1)
    X = data.drop('Id', axis=1)
    y = data[target_column]
    if 'sellPrice' in X.columns:
                    X = X.drop(columns=[target_column])


    # Get unique values for categorical features
    unique_suburbs = data['suburb'].unique().tolist()
    unique_postal_codes = data['postalCode'].unique().tolist()
    unique_prop_types = data['propType'].unique().tolist()

    # Get min and max values for numeric features
    min_bed = int(data['bed'].min())
    max_bed = int(data['bed'].max())
    min_bath = int(data['bath'].min())
    max_bath = int(data['bath'].max())
    min_car = int(data['car'].min())
    max_car = int(data['car'].max())

    # Split data into 80% train and 20% test
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Decision Tree Regressor
    dtr = DecisionTreeRegressor(
        max_depth=max_depth, 
        min_samples_split=min_samples_split, 
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    
    dtr.fit(train_x, train_y)

    # Make predictions on the test set
    predictions = dtr.predict(test_x)

    # Calculate evaluation metrics
    mae = mean_absolute_error(test_y, predictions)
    mse = mean_squared_error(test_y, predictions)
    r2 = r2_score(test_y, predictions)

    # Save the model
    with open('Saved_models/decision_tree_regressor.pkl', 'wb') as model_file:
        pickle.dump(dtr, model_file)

    return test_x, test_y, mse, mae, r2, unique_suburbs, unique_postal_codes, unique_prop_types, (min_bed, max_bed), (min_bath, max_bath), (min_car, max_car), "Model trained successfully!"
