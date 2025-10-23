import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load the dataset
path = "C:\\Users\\mason\\OneDrive\\Desktop\\MATH_3310\\project_1_crime\\MATH3310_FA_2025_01_01_RALLS\\data\\raw\\crime.csv"
df = pd.read_csv(path)

# remove any non-numeric variables that are not considered socioeconomic or demographic variable
df = df.drop(columns=["state", "region"])

print(df.info())

# Normalize column names so they are valid Python identifiers for patsy
# (replace dots and spaces with underscores)
df.rename(columns=lambda c: c.replace('.', '_').replace(' ', '_'), inplace=True)

df.head()

