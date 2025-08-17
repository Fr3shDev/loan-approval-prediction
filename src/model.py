import pandas as pd

data = pd.read_csv('data/loan_approval_dataset.csv')

print(data.head())
print('Shape:', data.shape)
print(data.dtypes)
print(data.isna().sum())

#