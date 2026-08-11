import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score
)

# Load dataset
df = pd.read_csv("data.csv")

# Remove useless column
df = df.drop("Unnamed: 32", axis=1)

# Remove ID column
df = df.drop("id", axis=1)

# Convert diagnosis to numbers
df["diagnosis"] = df["diagnosis"].map({
    "M": 1,
    "B": 0
})

print(df.head())

print("\nShape:")
print(df.shape)
# Features
X = df.drop("diagnosis", axis=1)

# Target
y = df["diagnosis"]

print("\nX Shape:")
print(X.shape)

print("\ny Shape:")
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data:")
print(X_train.shape)

print("\nTesting Data:")
print(X_test.shape)


models = {
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier()
}

results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    mcc = matthews_corrcoef(y_test, predictions)
    auc = roc_auc_score(y_test, predictions)

    results.append([
        name,
        accuracy,
        auc,
        precision,
        recall,
        f1,
        mcc
    ])

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1",
        "MCC"
    ]
)

results_df.to_csv(
    "metrics.csv",
    index=False
)

print(results_df)