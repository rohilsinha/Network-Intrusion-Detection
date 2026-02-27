
# Step 1: Import required libraries


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



# Step 2: Load the dataset


data = pd.read_csv("dataset/nsl_kdd.csv")

# FIX: clean column names
data.columns = data.columns.str.replace("'", "", regex=False).str.strip()

print("First 5 rows of dataset:")
print(data.head())

print("\nDataset shape:", data.shape)
print("\nColumn names in dataset:")
print(data.columns)


# Step 3: Separate and clean target label (FINAL FIX)

df = data.copy()

df["class"] = (
    df["class"]
    .astype(str)
    .str.lower()
    .str.strip()
    .str.replace(r"[^a-z]", "", regex=True)
)

print("\nUnique values in class column after cleaning:")
print(df["class"].unique())


# Binary labels
y = df["class"].apply(lambda x: 0 if x == "normal" else 1)

print("\nLabel distribution in full dataset:")
print(y.value_counts())


# Step 4: Encode categorical feature columns


X = df.drop("class", axis=1)

encoder = LabelEncoder()
for col in X.columns:
    if X[col].dtype == "object":
        X[col] = encoder.fit_transform(X[col])




# Step 5: Split data into training and testing


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)




# Step 6: Train Naïve Bayes model


model = GaussianNB()
model.fit(X_train, y_train)



# Step 7: Make predictions


y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)



# Step 8: Evaluate model performance


print("\nModel Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))



# Step 9: Real-time intrusion detection simulation


print("\n--- Real-Time Intrusion Detection Simulation ---")

sample_packet = X_test.iloc[[0]]


probability = model.predict_proba(sample_packet)
prediction = model.predict(sample_packet)

print("Intrusion Probability [Normal, Attack]:", probability)

if prediction[0] == 1:
    print("Prediction: ATTACK detected!")
else:
    print("Prediction: NORMAL traffic")



# Step 10: Plot confusion matrix


plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Intrusion Detection")
plt.show()
