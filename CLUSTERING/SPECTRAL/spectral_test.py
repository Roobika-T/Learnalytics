# test.py
import numpy as np
from sklearn.neighbors import NearestNeighbors
from CLUSTERING.SPECTRAL.spectral_train import generate_data, spectral_clustering_from_scratch

def predict_cluster(new_point, X, labels):
    nbrs = NearestNeighbors(n_neighbors=1).fit(X)
    distance, index = nbrs.kneighbors([new_point])
    return labels[index[0][0]]

if __name__ == '__main__':
    # Generate data and perform clustering
    X, _ = generate_data('circles', 1000)
    labels, _ = spectral_clustering_from_scratch(X)

    # Test with a new point
    new_point = np.array([0.5, -0.1])  # Example point
    cluster = predict_cluster(new_point, X, labels)
    print(f"The new point {new_point} is assigned to cluster: {cluster}")
