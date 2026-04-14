import pandas as pd

# Load dataset
train = pd.read_csv("data/titanic.csv")

print(train.head())
print(train.shape)
print(train.info())
print(train.isnull().sum())
