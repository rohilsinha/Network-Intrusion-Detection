from scipy.io import arff
import pandas as pd

# Load ARFF file
data, meta = arff.loadarff("dataset/KDDTest+.arff")

# Convert to DataFrame
df = pd.DataFrame(data)

# Decode byte strings to normal strings
for column in df.select_dtypes([object]).columns:
    df[column] = df[column].str.decode('utf-8')

# Save as CSV
df.to_csv("dataset/nsl_kdd.csv", index=False)

print("Conversion successful: dataset/nsl_kdd.csv created")
