import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Load data
data = pd.read_csv('data/loan_approval_dataset.csv')

# Clean column names by stripping whitespace
data.columns = data.columns.str.strip()

print(data.head())
print('Shape:', data.shape)
print("Cleaned column names:", list(data.columns))
print(data.dtypes)
print(data.isna().sum())

# Fix the pandas FutureWarning by using proper syntax
# Fill missing numerical columns with median
num_cols = data.select_dtypes(include=['float64', 'int64']).columns
for col in num_cols:
    data[col] = data[col].fillna(data[col].median())

# Fill missing categorical columns with mode (most frequent value)
cat_cols = data.select_dtypes(include=['object']).columns
for col in cat_cols: 
    data[col] = data[col].fillna(data[col].mode()[0])

# Separate target variable before encoding
target_col = 'loan_status'  # Based on your output, this is the correct column name
X = data.drop(target_col, axis=1)
y = data[target_col].str.strip()  # Remove any whitespace from target values

# Convert categorical features to numeric using one-hot encoding
X_encoded = pd.get_dummies(X, drop_first=True)

# Convert target to binary (check for exact values first)
print(f"Unique target values: {y.unique()}")
print(f"Target value counts:\n{y.value_counts()}")

# Convert to binary - handle case variations
y_binary = (y.str.lower() == 'approved').astype(int)

print(f"\nFeature columns after encoding: {list(X_encoded.columns)}")
print(f"Original target distribution:\n{y.value_counts()}")
print(f"Binary target distribution:\n{pd.Series(y_binary).value_counts()}")

# Check for any issues with the target conversion
print(f"Unique values in original target: {y.unique()}")
print(f"Unique values in binary target: {y_binary.unique()}")

# Split the data with error handling
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )
    print("\nUsing stratified split")
except ValueError as e:
    print(f"Stratified split failed: {e}")
    print("Using regular split instead")
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_binary, test_size=0.2, random_state=42
    )

print(f"\nTraining set target distribution:\n{pd.Series(y_train).value_counts()}")
print(f"Test set target distribution:\n{pd.Series(y_test).value_counts()}")

# Train models
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Logistic Regression evaluation
y_pred_lr = lr.predict(X_test)
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, y_pred_lr))

# Decision Tree evaluation
y_pred_dt = dt.predict(X_test)
print("\nDecision Tree Classification Report:")
print(classification_report(y_test, y_pred_dt))

# Address class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# Retrain logistic regression on balanced data
lr_sm = LogisticRegression(max_iter=1000, random_state=42)
lr_sm.fit(X_train_sm, y_train_sm)

# Evaluate again
y_pred_sm = lr_sm.predict(X_test)
print("\nLogistic Regression after SMOTE Classification Report:")
print(classification_report(y_test, y_pred_sm))