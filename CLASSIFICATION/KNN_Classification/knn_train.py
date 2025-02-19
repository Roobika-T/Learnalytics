import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

class KNNClassifier:
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors)

    def train_model(self, X, y):
        self.model.fit(X, y)

    def save_model(self, filename):
        with open(filename, 'wb') as model_file:
            pickle.dump(self.model, model_file)

# Data preprocessing
def load_iris_data(filename):
    # Load the dataset
    data = pd.read_csv(filename)
    # Separate features and target variable
    X = data.drop('species', axis=1)
    y = data['species']
    return X, y

# Training process
def main():
    # Load the dataset
    X, y = load_iris_data('Datasets/IRIS.csv')
    
    # Split the dataset into training and testing sets (optional)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Optional: Scale the data if necessary
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Create KNN Classifier
    knn = KNNClassifier(n_neighbors=3)  # You can adjust the number of neighbors as needed
    knn.train_model(X_train_scaled, y_train)

    # Save the trained model
    knn.save_model('Saved_models/knn_model.pkl')
    print("KNN model saved successfully.")

if __name__ == "__main__":
    main()