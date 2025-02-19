**ML Playground**

Welcome to the **ML Playground**! This project provides an interactive interface for exploring a range of popular machine learning algorithms for regression, classification, and clustering tasks. 

Built using **Streamlit** for the frontend and **scikit-learn** for model development, this project makes it easy to visualize and understand different ML models.

The project Working has been deployed in: 
```bash
https://mlplayground-3pgerzuzdwge7jzl5hycud.streamlit.app/
```
## 📋 Table of Contents
- [Introduction](#-introduction)
- [Features](#-features)
- [Getting Started](#-getting-started)
- [Installation](#-installation)
- [Procedure](#-procedure)
- [Usage](#-usage)
- [Model Types](#-model-types)
- [Contributing](#-contributing)

## 👋 Introduction

The **ML Playground** project is an interactive application where users can train and test various machine learning models with customizable datasets. Using **Streamlit** as the interface, users can choose models, train them, and evaluate results with metrics, visualizations, and model comparisons.

## ✨ Features
- Supports **Regression**, **Classification**, and **Clustering** models
- Includes 15+ machine learning models, from **Linear Regression** to **Neural Networks**
- **Streamlit UI** for model selection, training, and testing
- Options to choose and upload various datasets for different model types
- Visual representations and evaluation metrics for easy model comparison

## 🚀 Getting Started
1. **Clone the repository**:
    ```bash
    git clone https://github.com/22pt16/ML_Playground.git
    ```

2. **Navigate to the project directory**:
    ```bash
    cd ML_Playground
    ```

## 🛠️ Installation

To set up the environment, install the necessary libraries:

```bash
pip install streamlit scikit-learn pandas numpy matplotlib seaborn
```

**Verify your installation**:

```bash
python -c "import streamlit as st; print(st.__version__)"
python -c "import sklearn; print(sklearn.__version__)"
python -c "import pandas as pd; print(pd.__version__)"
python -c "import numpy as np; print(np.__version__)"
```

## ▶️ Usage

Run the application from the terminal:

```bash
streamlit run streamlit.py
```

- **Select a Model**: Choose from regression, classification, and clustering models.
- **Load Dataset**: Upload or select a dataset, and the data will be preprocessed automatically.
- **Evaluate**: See metrics, visualizations, and insights for each model.

## 🧠 Model Types

### Regression Models
- **Linear Regression**
- **Multiple Regression**
- **Gradient Boosting (Regression)**
- **Decision Tree Regression**

### Classification Models
- **Logistic Regression**
- **Neural Network**
- **K-Nearest Neighbors (KNN)**
- **Naive Bayes**
- **Support Vector Machine (SVM)**
- **Decision Tree Classifier**

### Clustering Models
- **K-Means Clustering**
- **Spectral Clustering**
- **DBSCAN Clustering**
- **K-Medoids Clustering**
- **Mixture of Gaussians**
- **Principal Component Analysis (PCA)**


Thank you for exploring the **ML Playground**! If you have questions or feedback, feel free to open an issue. Enjoy exploring machine learning! 🧑‍💻
