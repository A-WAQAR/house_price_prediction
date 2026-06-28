
# load dataset
import pandas as pd
df = pd.read_csv("data.csv")
print(df)
x = df[["area"]]
y = df["price"]

# train-test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)

# Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
lr = LinearRegression()
lr.fit(x_train, y_train)
lr_pred = lr.predict(x_test)
lr_score = r2_score(
    y_test,
    lr_pred
)
print("linear regression:", lr_score)

# decision tree
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(random_state=42)
dt.fit(x_train, y_train)
dt_pred = dt.predict(x_test)
dt_score = r2_score(
    y_test,
    dt_pred
)
print("decision tree regressor:", dt_score)

# Random Forest
from  sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state=42)
rf.fit(x_train, y_train)
rf_pred = rf.predict(x_test)
rf_score = r2_score(
    y_test,
    rf_pred
)
print("random forest regressor:", rf_score)

# compare score
print("lr:", lr_score)
print("dtr:", dt_score)
print("rfr:", rf_score)

# best model selection
scores = {
    "Linear Regression": lr_score,
    "Decision Tree": dt_score,
    "Random Forest": rf_score
}
best_model = max(
    scores,
    key=scores.get
)
print("best_model:", best_model)

# save best model
import joblib 
joblib.dump(
    lr,
    "best_model.pkl"
)

