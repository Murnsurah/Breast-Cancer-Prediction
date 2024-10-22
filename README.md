# Breast-Cancer-Prediction Using Machine Learning
## Introduction
Breast cancer is one of the most common cancers among women worldwide. Early detection is crucial for effective treatment and can significantly improve survival rates. This project aims to develop a machine learning model to predict whether a tumor is benign or malignant based on various features.

## Technologies Used
- Python
- Scikit-learn
- Pandas
- NumPy
- Streamlit

## Dataset
The dataset used in this project is sourced from the **Wisconsin Breast Cancer Dataset**. It contains numerous records of tumors along with various features that describe their characteristics. Each record has 30 features, with one categorical target indicating whether the tumor is benign or malignant.

# Exploratory Data Analysis
Exploratory Data Analysis (EDA) is a critical step in the data science workflow, helping to uncover insights and inform modeling decisions. In this project, EDA was performed on the Wisconsin Breast Cancer dataset to achieve the following:

- **Understanding the Dataset:** The dataset comprises 569 records with 30 features each. Key features include radius, texture, and various symmetry measures.

- **Data Visualization:** Various visualizations were created to understand the distribution of features and the relationship between them. Common visualizations included:
  - Histograms to visualize the distribution of individual features.
  - Box plots to identify outliers in numerical features.
  - Correlation heatmaps to identify relationships between features.

## Model Training
1. **Data Preprocessing:**
   - Normalizing or standardizing numerical features as necessary.
   - Splitting the dataset into training and testing sets.

2. **Model Selection:**
   - A **Random Forest Classifier** was chosen due to its robustness and ability to handle high-dimensional data effectively.
   - Hyperparameters were tuned using cross-validation to optimize model performance.

3.  **Performance Evaluation:**
   - Model accuracy, precision, recall, and F1-score were calculated on the test set to evaluate the model's effectiveness

