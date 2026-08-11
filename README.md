Logistic Regression

Logistic Regression provided strong classification performance with 95.61% accuracy and balanced precision and recall. It performed reliably on the breast cancer dataset.

Decision Tree

Decision Tree achieved acceptable results but produced the lowest performance among the implemented models. It may have slightly overfitted the training data.

KNN

KNN achieved similar accuracy to Logistic Regression and produced perfect precision, indicating very few false positives.

Naive Bayes

Naive Bayes produced the highest accuracy, MCC, and AUC scores among all models. It generalized extremely well on this dataset.

Random Forest

Random Forest performed strongly and provided stable predictions with high accuracy and F1 score. Ensemble learning improved robustness.

Overall Winner

Naive Bayes was the best-performing model on this dataset because it achieved the highest Accuracy (97.37%), AUC (96.51%), F1 Score (96.39%), and MCC (94.47%).

# Machine Learning Assignment 2

## Problem Statement

Build and evaluate multiple machine learning classification models on a public dataset and deploy them using Streamlit.

## Dataset Description

Dataset: Breast Cancer Wisconsin Dataset

Instances: 569

Features: 30

Target Variable:
- M = Malignant
- B = Benign

Source:
https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

## Models Used

1. Logistic Regression
2. Decision Tree
3. KNN
4. Naive Bayes
5. Random Forest

## Evaluation Results

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------|---------|---------|---------|---------|---------|---------|
| Logistic Regression | 0.9561 | 0.9464 | 0.9750 | 0.9070 | 0.9398 | 0.9068 |
| Decision Tree | 0.9386 | 0.9369 | 0.9091 | 0.9302 | 0.9195 | 0.8701 |
| KNN | 0.9561 | 0.9419 | 1.0000 | 0.8837 | 0.9383 | 0.9086 |
| Naive Bayes | 0.9737 | 0.9651 | 1.0000 | 0.9302 | 0.9639 | 0.9447 |
| Random Forest | 0.9649 | 0.9581 | 0.9756 | 0.9302 | 0.9524 | 0.9253 |

## Observations

### Logistic Regression
Provided strong overall performance and balanced classification results.

### Decision Tree
Performed reasonably well but had the lowest overall performance among the models.

### KNN
Achieved very high precision with competitive accuracy.

### Naive Bayes
Produced the best overall results with the highest accuracy and MCC score.

### Random Forest
Delivered stable and highly accurate predictions through ensemble learning.

## Best Performing Model

Naive Bayes

Accuracy: 97.37%

## Streamlit App Link

(Add after deployment)

## GitHub Repository Link

(Add after GitHub upload)