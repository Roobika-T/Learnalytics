# train.py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
import pickle

def generate_synthetic_data(n_samples, n_features, n_clusters):
    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=42)
    return X, y

def pca_from_scratch(X, n_components):
    # Calculate the mean of each feature
    X_meaned = X - np.mean(X, axis=0)
    cov_matrix = np.cov(X_meaned, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sorted_idx = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_idx]
    eigenvector_subset = sorted_eigenvectors[:, :n_components]
    X_reduced = np.dot(X_meaned, eigenvector_subset)
    
    # Return the reduced data, eigenvectors, and the mean used for centering
    return X_reduced, eigenvector_subset, np.mean(X, axis=0)

def train_pca_kmeans(X, n_clusters, n_components):
    X_reduced, eigenvector_subset, mean_vector = pca_from_scratch(X, n_components)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_reduced)
    silhouette = silhouette_score(X_reduced, labels)
    
    # Save models
    with open("Saved_models/pca_kmeans_model.pkl", "wb") as f:
        pickle.dump((eigenvector_subset, mean_vector, kmeans), f)
    
    return labels, X_reduced, silhouette

def load_models():
    with open("Saved_models/pca_kmeans_model.pkl", "rb") as f:
        eigenvector_subset, mean_vector, kmeans = pickle.load(f)
    return eigenvector_subset, mean_vector, kmeans
