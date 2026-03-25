# Step 1: Import required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score


# Step 2: Load the dataset

data = pd.read_csv("dataset/nsl_kdd.csv")

# Clean column names
data.columns = data.columns.str.replace("'", "", regex=False).str.strip()

print("First 5 rows of dataset:")
print(data.head())

print("\nDataset shape:", data.shape)


# Step 3: Clean target label

df = data.copy()

df["class"] = (
    df["class"]
    .astype(str)
    .str.lower()
    .str.strip()
    .str.replace(r"[^a-z]", "", regex=True)
)

print("\nUnique values in class column:")
print(df["class"].unique())


# Binary classification (Normal = 0, Attack = 1)
y = df["class"].apply(lambda x: 0 if x == "normal" else 1)

print("\nLabel distribution:")
print(y.value_counts())


# Step 4: Encode categorical features

X = df.drop("class", axis=1)

encoder = LabelEncoder()
for col in X.columns:
    if X[col].dtype == "object":
        X[col] = encoder.fit_transform(X[col])


# Step 5: Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# MODEL 1: NAIVE BAYES

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

nb_pred = nb_model.predict(X_test)


# MODEL 2: RANDOM FOREST


rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)


# EVALUATION


print("\n==============================")
print(" NAIVE BAYES RESULTS")
print("==============================")

print("Accuracy:", accuracy_score(y_test, nb_pred))
print("Precision:", precision_score(y_test, nb_pred))
print("Recall:", recall_score(y_test, nb_pred))
print("F1 Score:", f1_score(y_test, nb_pred))

print("\nConfusion Matrix:")
nb_cm = confusion_matrix(y_test, nb_pred)
print(nb_cm)

print("\nClassification Report:")
print(classification_report(y_test, nb_pred))


print("\n==============================")
print(" RANDOM FOREST RESULTS")
print("==============================")

print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Precision:", precision_score(y_test, rf_pred))
print("Recall:", recall_score(y_test, rf_pred))
print("F1 Score:", f1_score(y_test, rf_pred))

print("\nConfusion Matrix:")
rf_cm = confusion_matrix(y_test, rf_pred)
print(rf_cm)

print("\nClassification Report:")
print(classification_report(y_test, rf_pred))


# COMPARISON GRAPH


models = ["Naive Bayes", "Random Forest"]

accuracies = [
    accuracy_score(y_test, nb_pred),
    accuracy_score(y_test, rf_pred)
]

precisions = [
    precision_score(y_test, nb_pred),
    precision_score(y_test, rf_pred)
]

recalls = [
    recall_score(y_test, nb_pred),
    recall_score(y_test, rf_pred)
]

f1_scores = [
    f1_score(y_test, nb_pred),
    f1_score(y_test, rf_pred)
]


# Accuracy comparison
plt.figure()
plt.bar(models, accuracies)
plt.title("Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()


# F1 Score comparison (important metric)
plt.figure()
plt.bar(models, f1_scores)
plt.title("F1 Score Comparison")
plt.ylabel("F1 Score")
plt.show()


# CONFUSION MATRIX PLOTS


plt.figure(figsize=(5, 4))
sns.heatmap(nb_cm, annot=True, fmt='d', cmap='Blues')
plt.title("Naive Bayes Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

plt.figure(figsize=(5, 4))
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Greens')
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# REAL-TIME SIMULATION


print("\n--- Real-Time Intrusion Detection ---")

sample_packet = X_test.iloc[[0]]

nb_result = nb_model.predict(sample_packet)
rf_result = rf_model.predict(sample_packet)

print("Naive Bayes Prediction:",
      "ATTACK" if nb_result[0] == 1 else "NORMAL")

print("Random Forest Prediction:",
      "ATTACK" if rf_result[0] == 1 else "NORMAL")