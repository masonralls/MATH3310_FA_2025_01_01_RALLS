import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load the dataset
path = "C:\\Users\\mason\\OneDrive\\Desktop\\MATH_3310\\project_1_crime\\MATH3310_FA_2025_01_01_RALLS\\data\\raw\\crime.csv"
df = pd.read_csv(path)
df.head()

# Display basic information about the dataset
print(df.info())

# Check for missing values
print(df.isnull().sum())