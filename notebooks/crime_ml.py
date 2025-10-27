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

# Normalize column names so they are valid Python identifiers for patsy
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

# --- Preprocessing ---
numeric_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
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

# Train/test split for
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit on training data only
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
scores = cross_val_score(pipe, X_train, y_train, cv=kf, scoring='r2')
print("CV R² per fold:", scores)
print("Mean R²:", scores.mean())
print("Std dev:", scores.std())


