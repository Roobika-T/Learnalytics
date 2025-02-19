# dbscan_train.py
import pandas as pd
import pickle
import numpy as np
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def generate_data(shape='blobs', n_samples=100, noise=0.1):
    if shape == 'blobs':
        data, _ = make_blobs(n_samples=n_samples, centers=3, random_state=42)
    elif shape == 'moons':
        data, _ = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    elif shape == 'circles':
        data, _ = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)
    else:
        raise ValueError("Shape not recognized. Choose 'blobs', 'moons', or 'circles'.")
    return data

def apply_pca(data, n_components=2):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(data)

def train_dbscan(shape='blobs', n_samples=100, eps=0.5, min_samples=5, noise=0.1):
    data = generate_data(shape, n_samples, noise)
    data_pca = apply_pca(data)

    # Train DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(data_pca)

    silhouette = silhouette_score(data_pca, clusters) if len(set(clusters)) > 1 else -1

    # Save the model
    model_path = 'Saved_models/dbscan_model.pkl'
    with open(model_path, 'wb') as model_file:
        pickle.dump({'model': dbscan, 'clusters': clusters, 'pca': PCA(n_components=2).fit(data)}, model_file)

    # Plot the clusters
    plt.figure(figsize=(10, 6))
    plt.scatter(data_pca[:, 0], data_pca[:, 1], c=clusters, cmap='viridis', marker='o')
    plt.title(f'DBSCAN Clustering Results - {shape.capitalize()}')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(label='Cluster Label')
    plt.savefig('Saved_models/dbscan_plot.png')  # Save the plot
    plt.close()

    return f"DBSCAN model trained with silhouette score: {silhouette:.4f}", silhouette, data_pca, clusters
