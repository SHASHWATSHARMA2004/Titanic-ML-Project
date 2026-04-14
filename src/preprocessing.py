import pandas as pd
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv("data/titanic.csv")

train["Age"] = train["Age"].fillna(train["Age"].median())
train["Embarked"] = train["Embarked"].fillna(train["Embarked"].mode()[0])
train = train.drop("Cabin", axis=1)

encoder = LabelEncoder()
train["Sex"] = encoder.fit_transform(train["Sex"])
train["Embarked"] = encoder.fit_transform(train["Embarked"])

print(train.head())