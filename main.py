from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

model = joblib.load('titanic_model.pkl')

app = FastAPI()

class Passenger(BaseModel):
    Pclass: int
    Sex: int
    Ageband: int
    Fareband: int
    Embarked: int
    faimily_size: int
    IsAlone: int
    Title: int

@app.post("/predict")
def predict(passenger: Passenger):
    data = np.array([[passenger.Pclass, passenger.Sex, passenger.Ageband, passenger.Fareband,
                      passenger.Embarked, passenger.faimily_size, passenger.IsAlone, passenger.Title]])
    prediction = model.predict(data)[0]
    result = "Survived" if prediction == 1 else "Died"
    return {"prediction": int(prediction), "result": result}
