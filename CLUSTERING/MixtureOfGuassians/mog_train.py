import numpy as np
from sklearn.datasets import make_moons, make_circles
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

def generate_data(data_type='moons', n_samples=1000):
    if data_type == 'moons':
        X, y = make_moons(n_samples=n_samples, noise=0.1)
    elif data_type == 'circles':
        X, y = make_circles(n_samples=n_samples, noise=0.1)
    else:
        raise ValueError("data_type must be 'moons' or 'circles'")
    return X

def train_mog(n_clusters=2, data_type='moons', n_samples=1000):
    # Step 1: Generate synthetic data
    X = generate_data(data_type, n_samples)

    # Step 2: Fit a Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm.fit(X)
    
    # Step 3: Get cluster labels
    labels = gmm.predict(X)
    
    # Calculate silhouette score
    silhouette = silhouette_score(X, labels)
    
    return labels, silhouette, X

if __name__ == '__main__':
    labels, silhouette, X = train_mog(n_clusters=2, data_type='moons', n_samples=1000)
    print("Silhouette Score:", silhouette)