# train.py
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score

# Load Titanic dataset
def load_data():
    data = sns.load_dataset('titanic').dropna(subset=['age', 'fare', 'sex', 'pclass', 'survived'])
    data['sex'] = data['sex'].apply(lambda x: 1 if x == 'male' else 0)
    X = data[['pclass', 'sex', 'age', 'fare']].values
    y = data['survived'].values
    return X, y

# Calculate Gini impurity
def gini(y):
    counts = Counter(y)
    impurity = 1 - sum((count / len(y)) ** 2 for count in counts.values())
    return impurity

# Calculate Entropy
def entropy(y):
    counts = Counter(y)
    total = len(y)
    return -sum((count / total) * np.log2(count / total) for count in counts.values() if count)

# Determine the best split based on Gini or Entropy
def best_split(X, y, criterion):
    best_gain = 0
    best_split = None
    current_impurity = gini(y) if criterion == "gini" else entropy(y)

    for col in range(X.shape[1]):
        thresholds = np.unique(X[:, col])
        for threshold in thresholds:
            left = y[X[:, col] <= threshold]
            right = y[X[:, col] > threshold]
            if len(left) == 0 or len(right) == 0:
                continue

            p_left, p_right = len(left) / len(y), len(right) / len(y)
            gain = current_impurity - (p_left * (gini(left) if criterion == "gini" else entropy(left)) +
                                       p_right * (gini(right) if criterion == "gini" else entropy(right)))

            if gain > best_gain:
                best_gain = gain
                best_split = (col, threshold)

    return best_split, best_gain

# Node class for decision tree structure
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

# Build the decision tree
def build_tree(X, y, depth=0, max_depth=5, criterion="gini"):
    if len(set(y)) == 1 or depth == max_depth:
        return Node(value=Counter(y).most_common(1)[0][0])

    feature, threshold, gain = None, None, None
    split, gain = best_split(X, y, criterion)
    if gain == 0:
        return Node(value=Counter(y).most_common(1)[0][0])

    feature, threshold = split
    left_mask = X[:, feature] <= threshold
    right_mask = X[:, feature] > threshold

    left = build_tree(X[left_mask], y[left_mask], depth + 1, max_depth, criterion)
    right = build_tree(X[right_mask], y[right_mask], depth + 1, max_depth, criterion)

    return Node(feature, threshold, left, right)

# Make predictions with the decision tree
def predict_tree(node, x):
    if node.value is not None:
        return node.value
    if x[node.feature] <= node.threshold:
        return predict_tree(node.left, x)
    else:
        return predict_tree(node.right, x)

# Fit the model
def fit(X, y, max_depth=5, criterion="gini"):
    tree = build_tree(X, y, max_depth=max_depth, criterion=criterion)
    # Save model using pickle
    with open('Saved_models/decision_tree_model.pkl', 'wb') as f:
        pickle.dump(tree, f)
    return tree

# Predict using the model
def predict(model, X):
    return np.array([predict_tree(model, x) for x in X])

# Accuracy metric
def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

# Precision, recall, and F1 metrics
def calculate_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return precision, recall, f1
