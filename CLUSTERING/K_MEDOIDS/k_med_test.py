import pandas as pd
import pickle
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

def load_model(model_path='Saved_models/k_medoids_model.pkl'):
    with open(model_path, 'rb') as model_file:
        return pickle.load(model_file)

def predict_cluster(new_data, model_path='Saved_models/k_medoids_model.pkl'):
    model = load_model(model_path)

    new_data_df = pd.DataFrame([new_data], columns=['GDP per capita', 'Social support', 'Healthy life expectancy', 'Freedom to make life choices'])

    # Reuse the PCA instance
    pca = model['pca']
    new_data_pca = pca.transform(new_data_df)  # Use transform instead of fit_transform

    distances = cdist(new_data_pca, model['medoids'], metric='euclidean')
    return np.argmin(distances)
