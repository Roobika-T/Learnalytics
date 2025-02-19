import pandas as pd
import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

def load_data(file_path='Datasets/k_med_happiness_report.csv'):
    return pd.read_csv(file_path)



def initialize_medoids(data, n_clusters):
    indices = np.random.choice(len(data), n_clusters, replace=False)
    return data[indices]

def assign_clusters(data, medoids, metric):
    distances = cdist(data, medoids, metric=metric)
    return np.argmin(distances, axis=1)

def compute_cost(data, clusters, medoids, metric):
    total_cost = 0
    for cluster in np.unique(clusters):
        cluster_data = data[clusters == cluster]
        total_cost += np.sum(cdist(cluster_data, [medoids[cluster]], metric=metric))
    return total_cost

def update_medoids(data, clusters, n_clusters, metric):
    new_medoids = []
    for cluster in range(n_clusters):
        cluster_data = data[clusters == cluster]
        distances = cdist(cluster_data, cluster_data, metric=metric)
        medoid_index = np.argmin(distances.sum(axis=1))
        new_medoids.append(cluster_data[medoid_index])
    return np.array(new_medoids)

def train_kmedoids(n_clusters=3, max_iter=300, metric="euclidean", data_path='Datasets/k_med_happiness_report.csv'):

    data = load_data(data_path)
    # Select relevant columns from the dataset
    data = data[['GDP per capita', 'Social support', 'Healthy life expectancy', 'Freedom to make life choices']]

    # Apply PCA and store the PCA object
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data)

    # Initialize medoids
    medoids = initialize_medoids(data_pca, n_clusters)
    for i in range(max_iter):
        # Assign clusters based on current medoids
        clusters = assign_clusters(data_pca, medoids, metric)
        # Update medoids
        new_medoids = update_medoids(data_pca, clusters, n_clusters, metric)

        # Stop if no improvement in medoids
        if np.array_equal(medoids, new_medoids):
            break
        else:
            medoids = new_medoids

    silhouette = silhouette_score(data_pca, clusters)

    model_path = 'Saved_models/k_medoids_model.pkl'
    with open(model_path, 'wb') as model_file:
        pickle.dump({'medoids': medoids, 'clusters': clusters, 'pca': pca}, model_file)

    return f"K-Medoids model trained with silhouette score: {silhouette:.4f}", silhouette, data_pca, clusters
