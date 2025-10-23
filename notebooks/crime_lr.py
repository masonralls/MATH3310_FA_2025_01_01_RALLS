import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# load the dataset
path = "C:\\Users\\mason\\OneDrive\\Desktop\\MATH_3310\\project_1_crime\\MATH3310_FA_2025_01_01_RALLS\\data\\raw\\crime.csv"
df = pd.read_csv(path)
df.head()

# Check for missing values
print(df.isnull().sum())
 
 # remove any non-numeric variables that are not considered socioeconomic or demographic variable
df = df.drop(columns=["state", "region"])

print(df.info())

# Normalize column names so they are valid Python identifiers for patsy
# (replace dots and spaces with underscores)
df.rename(columns=lambda c: c.replace('.', '_').replace(' ', '_'), inplace=True)

# Define features and target variable (use new name murder_rate)
X = df.drop('murder_rate', axis=1)
y = df['murder_rate']

# create a linear regression model
model = LinearRegression()

# fit the model
model.fit(X, y)

# print the R^2 score
print(f"Model R^2 score (sklearn): {model.score(X, y):.4f}")

# fit the full model
full_model = ols('murder_rate ~ poverty + high_school + college + single_parent + unemployed + metropolitan', data=df).fit()
print(full_model.summary())

# fit nested models
nested_model_1 = ols('murder_rate ~ poverty', data=df).fit()
nested_model_2 = ols('murder_rate ~ poverty + high_school', data=df).fit()
nested_model_3 = ols('murder_rate ~ poverty + high_school + college', data=df).fit()

# print the summary of each nested model
print(f"Summary for Nested Model 1:")
print(nested_model_1.summary())
print(f"Summary for Nested Model 2:")
print(nested_model_2.summary())
print(f"Summary for Nested Model 3:")
print(nested_model_3.summary())

# use anova_lm to compare the models with the full model
anova_results = anova_lm(nested_model_1, nested_model_2, nested_model_3, full_model)
print(anova_results)

# which nested models are statistically significant?
significant_models = anova_results[anova_results['Pr(>F)'] < 0.05]
print(f"The statistically significant models are: {significant_models.index.tolist()}")

# adding high_school to poverty alone significantly improved the model
# adding college to the model did not significantly improve the model
# adding single_parent, unemployed and metropolitan to the model significantly improved the model

# if our goal is prediction, the best model includes all variables except college

# make a new linear regression model using only the significant variables
best_model = ols('murder_rate ~ poverty + high_school + single_parent + unemployed + metropolitan', data=df).fit()
print(best_model.summary())

# print the R^2 score for the best model
print(f"Model R^2 score (statsmodels): {best_model.rsquared:.4f}")

# test for correlation between predictors
corr = df[['poverty', 'high_school', 'single_parent', 'unemployed', 'metropolitan']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# perform analysis to check the standard assumptions of linear regression

# 1. Linearity
sns.pairplot(df[['murder_rate', 'poverty', 'high_school', 'single_parent', 'unemployed', 'metropolitan']])
plt.show()

# 2. Homoscedasticity
residuals = best_model.resid
fitted = best_model.fittedvalues
sns.scatterplot(x=fitted, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs Fitted')
plt.show()

# 3. Normality of residuals
sns.histplot(residuals, kde=True)
plt.title('Residuals Distribution')
plt.show()


# Analysis of Assumptions: Summary

# Linearity looks acceptable. Some predictors might share information (see correlation matrix) that may effect coefficient stability but not the overall linear fit.
# No major violation of homoscedasticity.
# Residuals appear to be normally distributed. A slight skew is probably a result of the small sample size.
