import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv("data/titanic.csv")

print(train["Survived"].value_counts())
print(pd.crosstab(train["Sex"], train["Survived"]))
print(pd.crosstab(train["Pclass"], train["Survived"]))

train["Age"].hist()
plt.show()