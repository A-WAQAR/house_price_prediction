
# create dataset
import csv
data = [["area", "price"],
        [500, 5],
        [800, 8],
        [1000, 10],
        [1200, 12],
        [1500, 15],
        [1800, 18],
        [2000, 20],
        [2200, 22],
        [2500, 25],
        [3000, 30]
]
with open("data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(data)

# load dataset
import pandas as pd
df = pd.read_csv("data.csv")

# basic data analysis
print(df.shape)
print(df.info())
print(df.describe())

# features and target
x = df[["area"]]
y = df["price"]

# train/test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# train model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)

predictions = model.predict([[1600]])
print(predictions)

# evaluate model
from sklearn.metrics import r2_score
predx = model.predict(x_test)
score = r2_score(y_test, predx)
print(score)

# save model
import joblib
joblib.dump(model, "model.pkl")

