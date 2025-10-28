import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer,  make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score


# load the dataset
path = "C:\\Users\\mason\\OneDrive\\Desktop\\MATH_3310\\project_1_crime\\MATH3310_FA_2025_01_01_RALLS\\data\\raw\\crime.csv"
df = pd.read_csv(path)

# Check for missing values
print(df.isnull().sum())

# Normalize column names 
# (replace dots and spaces with underscores)
df.rename(columns=lambda c: c.replace('.', '_').replace(' ', '_'), inplace=True)
df.head()

# Drop non-features and define target
y = df['murder_rate']
X = df.drop(columns=['murder_rate', 'state'])

# Identify column types (numeric vs categorical)
numeric_features = selector(dtype_include=np.number)(X)

# If 'region' is the only categorical, we can also do: categorical_features = ['region']
categorical_features = [c for c in X.columns if c not in numeric_features]

print("numeric_features:", numeric_features)
print("categorical_features:", categorical_features)

# Create a Linear Regression model on the training data to compare with the cross validation results

# Train Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# encode categorical features with one-hot encoding

X_train_encoded = pd.get_dummies(X_train, columns=categorical_features, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, columns=categorical_features, drop_first=True)

# Align the train and test sets
X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

# Fit the linear regression model
lr = LinearRegression()
lr.fit(X_train_encoded, y_train)

# Evaluate on test set
y_pred_lr = lr.predict(X_test_encoded)
r2_train_test = r2_score(y_test, y_pred_lr)
print("Train-Test Split R²:", r2_train_test)

# Calculate Adjusted R² for train-test split model
n_total = X.shape[0]  # total number of samples
p_encoded = X_train_encoded.shape[1]  # number of features after encoding

# Use total sample size instead of just test set size for adjustment
adj_r2_train_test = 1 - (1 - r2_train_test) * (n_total - 1) / (n_total - p_encoded - 1)
print("Train-Test Split Adjusted R²:", adj_r2_train_test)

# Cross Validation model

# --- Preprocessing ---
numeric_pipe = Pipeline([
    ('scaler', StandardScaler())
])

categorical_pipe = Pipeline([
    ('ohe', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))
])

preprocess = ColumnTransformer([
    ('num', numeric_pipe, numeric_features),
    ('cat', categorical_pipe, categorical_features)
])

# --- Full pipeline (preprocess -> model) ---
pipe = Pipeline([
    ('preprocess', preprocess),
    ('lr', LinearRegression())
])

pipe

# Train/test split for
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit pipeline on training data only
pipe.fit(X_train, y_train)

# Predict and plot
y_pred = pipe.predict(X_test)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(lims, lims, 'k--', lw=2)
plt.xlabel('Actual Murder Rate')
plt.ylabel('Predicted Murder Rate')
plt.title('Actual vs Predicted Murder Rate')
plt.show()

# Cross-validation on raw X/y (the pipeline handles transforms inside each fold)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X, y, cv=kf, scoring='r2')
print("CV R² per fold:", scores)
print("Mean R²:", scores.mean())
print("Std dev:", scores.std())


# Compute adjusted R² with actual feature count from pipeline
n = X.shape[0]  # number of samples
p = X.shape[1]  # number of features
adj_r2 = 1 - (1 - scores.mean()) * (n - 1) / (n - p - 1)
print(f"\nAdjusted R² : {adj_r2:.4f}")
