import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix


# Load dataset

data = pd.read_csv("dataset/nsl_kdd.csv")


data.columns = data.columns.str.replace("'", "", regex=False).str.strip()

print(data.head())
print(data.shape)


# Target label cleaning and binary conversion

df = data.copy()

df["class"] = (
    df["class"]
    .astype(str)
    .str.lower()
    .str.strip()
)

# normal -> 0, anomaly -> 1
y = df["class"].apply(lambda x: 0 if x == "normal" else 1)
print(y.value_counts())


# Feature selection and encoding

X = df.drop("class", axis=1)

encoder = LabelEncoder()
for col in X.columns:
    if X[col].dtype == "object":
        X[col] = encoder.fit_transform(X[col])


# Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

 
# Naive Bayes model training

model = GaussianNB()
model.fit(X_train, y_train)


# Initial output 

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
