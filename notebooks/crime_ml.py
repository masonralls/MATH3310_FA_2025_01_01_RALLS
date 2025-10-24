import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, KFold

# load the dataset
path = "C:\\Users\\mason\\OneDrive\\Desktop\\MATH_3310\\project_1_crime\\MATH3310_FA_2025_01_01_RALLS\\data\\raw\\crime.csv"
df = pd.read_csv(path)
print(df.info())

# Normalize column names so they are valid Python identifiers for patsy
# (replace dots and spaces with underscores)
df.rename(columns=lambda c: c.replace('.', '_').replace(' ', '_'), inplace=True)
df.head()

# use sklearn to perform a train test split
X = df.drop('murder_rate', axis=1)
y = df['murder_rate']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   

# create a linear regression model on the training data
model = LinearRegression()
model.fit(X_train, y_train)

# Fit the model to the training data and then use it to make predictions on the testing set
y_pred = model.predict(X_test)

# create a scatter plot of the predicted vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual Murder Rate')
plt.ylabel('Predicted Murder Rate')
plt.title('Actual vs Predicted Murder Rate')
plt.show()


# Evaluate LinearRegression model using stratified 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

pipe = Pipeline([('scaler', StandardScaler()), ('lr', LinearRegression())])

pipe

np.random.seed(42)

cross_val_scores = cross_val_score(pipe, X_train, y_train, cv=kf, scoring='r2')
print("Cross-validation scores:", cross_val_scores)

# inspecting per fold errors
print("Mean R²:", np.mean(cross_val_scores))
print("Std deviation:", np.std(cross_val_scores))

