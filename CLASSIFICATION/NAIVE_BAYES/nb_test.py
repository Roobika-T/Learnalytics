import streamlit as st
import pickle
import re
from CLASSIFICATION.NAIVE_BAYES.nb_train import NaiveBayes  # Ensure this imports your training class

# Function to load the model and likelihood table from the pickle file
def load_model():
    with open('Saved_models/naive_bayes_model.pkl', 'rb') as model_file:
        data = pickle.load(model_file)
        return data['model'], data['likelihood_table']

# Function to classify emails using the loaded Naive Bayes model
def classify_email(email):
    model, likelihood_table = load_model()
    words = re.findall(r'\b\w+\b', email.lower())
    good_prob = 1.0
    spam_prob = 1.0

    for word in words:
        if word in likelihood_table:
            good_prob *= likelihood_table[word][0]
            spam_prob *= likelihood_table[word][1]
        

    # Normalize
    total_prob = good_prob + spam_prob
    good_prob /= total_prob
    spam_prob /= total_prob

    print(f"Good Email Probability: {good_prob:.4f}, Spam Email Probability: {spam_prob:.4f}")
    return f"Good Email Probability: {good_prob:.4f}, Spam Email Probability: {spam_prob:.4f}"


