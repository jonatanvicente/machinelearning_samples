import pandas as pd

# Sample dataset with PII
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 40],
    'Email': ['alice@example.com', 'bob@example.com', 'charlie@example.com', 'david@example.com'],
    'Salary': [50000, 60000, 70000, 80000]
}

df = pd.DataFrame(data)

# Remove PII columns
df_anonymized = df.drop(columns=['Name', 'Email'])

# Generalize the 'Age' column by binning ages into ranges
age_bins = [0, 20, 30, 40, 50]
age_labels = ['0-20', '21-30', '31-40', '41-50']
df_anonymized['Age'] = pd.cut(df_anonymized['Age'], bins=age_bins, labels=age_labels, right=False)

print(df_anonymized)