
# create app
from fastapi import FastAPI 
import joblib
app = FastAPI()
model = joblib.load("model.pkl")

# create home route
@app.get("/")
def home():
    return {
        "message": " house price prediction API"
    }

# create prediction route
@app.get("/predict/{area}")
def predict(area:int):
    prediction = model.predict([[area]])
    return {
        "predicted price": float(prediction[0])
    }

@app.get("/predict/{area}")
def predict(area:float):
    if area <= 0:
        return {"error":"area must be positive"}
    prediction = model.predict([[area]])
    return {
        "area":area,
        "predicted price" : float(prediction[0])
    }
