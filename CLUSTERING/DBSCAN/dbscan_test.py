# dbscan_test.py
import pickle
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

def load_model(model_path='Saved_models/dbscan_model.pkl'):
    with open(model_path, 'rb') as model_file:
        return pickle.load(model_file)

def predict_cluster(new_data, model_path='Saved_models/dbscan_model.pkl'):
    model = load_model(model_path)
    
    # Transform new data to match PCA dimensions
    pca = model['pca']
    new_data_pca = pca.transform(np.array(new_data).reshape(1, -1))

    # Assign cluster based on the model's clustering
    clusters = model['model'].labels_
    distances = cdist(new_data_pca, model['pca'].inverse_transform(model['model'].components_), metric='euclidean')    
    return clusters[np.argmin(distances)]
