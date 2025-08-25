import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

data = pd.read_csv('data/loan_approval_dataset.csv')

print(data.head())
print('Shape:', data.shape)
print(data.dtypes)
print(data.isna().sum())

# Example: fill missing numerical columns with median
num_cols = data.select_dtypes(include=['float64', 'int64']).columns
for col in num_cols:
    data[col].fillna(data[col].median(), inplace=True)

# Fill missing categorical columns with mode (most frequent value)
cat_cols = data.select_dtypes(include=['object']).columns
for col in cat_cols: 
    data[col].fillna(data[col].mode()[0], inplace=True)

#Convert categorical columns to numeric using one-hot encoding or label encoding
data_encoded = pd.get_dummies(data, drop_first=True)

# Define features and target
X = data_encoded.drop('Loan_Status_Y', axis=1)
y = data_encoded['Loan_Status_Y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

#Logistic Regression evaluation
y_pred_lr = lr.predict(X_test)
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_lr))

#Decision Tree evaluation
y_pred_dt = dt.predict(X_test)
print("Decision Tree Classification Report:")
print(classification_report(y_test, y_pred_dt))