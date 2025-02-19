# train.py
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_circles, make_moons
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh

def generate_data(data_type='circles', n_samples=1000):
    if data_type == 'circles':
        X, y = make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
    elif data_type == 'moons':
        X, y = make_moons(n_samples=n_samples, noise=0.1)
    else:
        raise ValueError("data_type must be 'circles' or 'moons'")
    return X, y

def rbf_kernel_similarity(X, gamma=1.0):
    """Compute the RBF (Gaussian) kernel similarity matrix."""
    pairwise_sq_dists = squareform(pdist(X, 'sqeuclidean'))
    similarity_matrix = np.exp(-gamma * pairwise_sq_dists)
    return similarity_matrix

def spectral_clustering_from_scratch(X, n_clusters=2, gamma=1.0):
    # Step 1: Compute the RBF kernel similarity matrix
    similarity_matrix = rbf_kernel_similarity(X, gamma=gamma)

    # Step 2: Compute the Degree matrix and Laplacian matrix
    degree_matrix = np.diag(np.sum(similarity_matrix, axis=1))
    laplacian_matrix = degree_matrix - similarity_matrix

    # Step 3: Compute eigenvalues and eigenvectors of the Laplacian
    eigvals, eigvecs = eigh(laplacian_matrix, degree_matrix, subset_by_index=[0, n_clusters - 1])
    
    # Step 4: Perform k-means clustering on the eigenvectors (embedding space)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(eigvecs)
    
    # Calculate silhouette score
    silhouette = silhouette_score(X, labels)
    
    return labels, silhouette

if __name__ == '__main__':
    # Example run for testing
    X, y = generate_data('circles')
    labels, silhouette = spectral_clustering_from_scratch(X, gamma=15)  # Higher gamma for tighter clusters
    print("Silhouette Score:", silhouette)
