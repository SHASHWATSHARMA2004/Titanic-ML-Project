import pandas as pd

train = pd.read_csv("data/titanic.csv")

train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
train["IsAlone"] = train["FamilySize"].apply(lambda x: 1 if x == 1 else 0)

train["AgeGroup"] = pd.cut(
    train["Age"],
    bins=[0, 12, 19, 35, 60, 100],
    labels=["Child", "Teen", "Adult", "Middle", "Senior"]
)

print(train[["FamilySize", "IsAlone", "AgeGroup"]].head())