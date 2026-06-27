
# create sample dataset
import pandas as pd

data = {
    "area":[500,800,None,1200,1500,50000],
    "bedrooms":[1,2,2,None,3,20],
    "price":[5,8,10,12,15,500]
}

df = pd.DataFrame(data)
print(df)

# check missing values
print(df.isnull().sum())

# fill missing values
df["area"] = df["area"].fillna(
    df["area"].mean()
)
df["bedrooms"] = df["bedrooms"].fillna(
    df["bedrooms"].mean()
)
print(df["area"])
print(df["bedrooms"])

# detect outliers
import seaborn as sns
import matplotlib.pyplot as plt
sns.boxplot(x=df["area"])
plt.show()

# remove outliers(IQR Method)
Q1 = df["area"].quantile(0.25)
Q3 = df["area"].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
df = df[
    (df["area"] >= lower) &
    (df["area"] <= upper)
]
print(df)

# categorical encoding
data = {
    "city":[
        "Patna",
        "Delhi",
        "Patna",
        "Mumbai"
    ]
}
df = pd.DataFrame(data)
print(df)

# convert 
df_encoded = pd.get_dummies(
    df,
    columns= ["city"]
)
print(df_encoded)

# feature scaling
from sklearn.preprocessing import StandardScaler
scaler  = StandardScaler()
df[["area"]] = scaler.fit_transform(df[["area"]])
print(df["area"])

# create new features
df["price_per_sqft"] = (
    df["price"] / df["area"]
)
print(df["price_per_sqft"])

# check final dataset
print(df.head())
print(df.info())