
# load dataset
import pandas as pd
df = pd.read_csv("data.csv")

# dataset shape
print(df.shape)

# dataset information
print(df.info())

# missing values
print(df.isnull().sum())

# statistical summary
print(df.describe())

# histogram
import matplotlib.pyplot as plt
df["area"].hist()
plt.show()

# boxplot
import seaborn as sns
sns.boxplot(
    x = df["area"]
)
plt.show()

# correlation
print(df.corr())

# correlation heatmap
import seaborn as sns
sns.heatmap(
    df.corr(),
    annot = True
)
plt.show()

# scatter plot
plt.scatter(df["area"], df["price"])
plt.xlabel("area")
plt.ylabel("price")
plt.show()