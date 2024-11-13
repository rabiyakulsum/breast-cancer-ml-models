# Breast Cancer Detection

## Project Overview
This project aims to predict the likelihood of breast cancer based on patient data. By utilizing multiple machine learning models, we aim to create a system that can assist in early detection, potentially improving patient outcomes. The dataset includes various features such as age, tumor stage, estrogen status, and other clinical indicators. We evaluate multiple classification models to determine the best approach for this prediction task.

## Project Structure
The project is organized into the following steps:
1. **Data Loading and Preprocessing**: Load the data, handle missing values, encode categorical variables, and prepare the dataset for modeling.
2. **Exploratory Data Analysis (EDA)**: Analyze and visualize relationships between features and the target variable.
3. **Model Training and Evaluation**: Train and evaluate several models to find the best performer in terms of accuracy and other metrics.
4. **Model Comparison**: Compare the performance of each model to select the best model for breast cancer detection.

## Dataset Description
The dataset is provided in CSV format and includes the following columns:

- **Age**: Age of the patient
- **Race**: Race of the patient
- **Marital Status**: Marital status of the patient
- **T Stage**: Tumor stage
- **N Stage**: Node stage
- **6th Stage**: Cancer stage (6th edition)
- **differentiate**: Degree of differentiation
- **Grade**: Cancer grade
- **A Stage**: Additional stage information
- **Tumor Size**: Size of the tumor
- **Estrogen Status**: Estrogen receptor status
- **Progesterone Status**: Progesterone receptor status
- **Regional Node Examined**: Number of regional nodes examined
- **Reginol Node Positive**: Number of positive regional nodes
- **Survival Months**: Survival months since diagnosis
- **Status**: Survival status (target variable)

## Goals
- **Analyze** the data to gain insights into factors affecting breast cancer prognosis.
- **Build and Compare** multiple classification models to predict survival status.
- **Evaluate** each model’s performance to select the most accurate one for breast cancer detection.

## Models Used
The following machine learning models are implemented for comparison:
1. **Decision Tree**: `DecisionTreeClassifier(criterion='entropy')`
2. **Logistic Regression**: `LogisticRegression()`
3. **Random Forest**: `RandomForestClassifier(n_estimators=10)`
4. **Gradient Boosting**: `GradientBoostingClassifier(n_estimators=100)`
5. **K-Nearest Neighbors (KNN)**: `KNeighborsClassifier(n_neighbors=3)`
6. **Naive Bayes**: `GaussianNB()`
7. **Support Vector Classifier (SVC)**: `SVC(kernel='linear')`

## Methodology
1. **Data Preprocessing**: Clean and preprocess the data, including handling missing values and encoding categorical variables.
2. **Exploratory Analysis**: Use visualizations and statistical analysis to explore relationships between features and the target variable.
3. **Model Training**: Train each model on the training set.
4. **Model Evaluation**: Evaluate each model on the test set using accuracy, precision, recall, and F1-score.
5. **Comparison**: Compare models to identify the best performer for predicting survival status.

## Dependencies
- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn (optional, for visualization)


