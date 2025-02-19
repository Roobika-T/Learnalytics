# test.py
import numpy as np
import pickle
from CLUSTERING.PCA.pca_train import generate_synthetic_data, load_models, pca_from_scratch
from sklearn.metrics import silhouette_score

def evaluate_model(X):
    eigenvector_subset, kmeans = load_models()
    
    # Apply PCA from scratch
    X_meaned = X - np.mean(X, axis=0)
    X_reduced = np.dot(X_meaned, eigenvector_subset)
    
    # Predict clusters
    labels = kmeans.predict(X_reduced)
    
    # Evaluate clustering
    silhouette = silhouette_score(X_reduced, labels)
    
    return labels, X_reduced, silhouette

if __name__ == "__main__":
    X_test, _ = generate_synthetic_data(n_samples=300, n_features=10, n_clusters=3)
    labels, X_reduced, silhouette = evaluate_model(X_test)
    
    print(f"Silhouette Score on Test Set: {silhouette:.2f}")
