import numpy as np
from sklearn.mixture import GaussianMixture
from CLUSTERING.MixtureOfGuassians.mog_train import train_mog  # Importing the training function

def predict_cluster(new_point, gmm):
    """Predict the cluster for a new data point using the trained GMM."""
    return gmm.predict([new_point])[0]

if __name__ == '__main__':
    # Train the model and get the GMM
    labels, silhouette, X = train_mog(n_clusters=2, data_type='moons', n_samples=1000)
    
    # Create the GMM for prediction
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(X)

    # Test with a new point
    new_point = np.array([0.5, 0.2])  # Example point
    cluster = predict_cluster(new_point, gmm)
    print(f"The new point {new_point} is assigned to cluster: {cluster}")