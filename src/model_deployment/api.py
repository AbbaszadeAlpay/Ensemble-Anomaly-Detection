from datetime import datetime

from fastapi import FastAPI
from pydantic import BaseModel

from src.model_deployment import model_predict


class Input(BaseModel):
    credit_card: int
    date: datetime
    transaction_dollar_amount: float
    Long: float
    Lat: float
    city: str
    state: str
    zipcode: int
    credit_card_limit: int


class PredictionOut(BaseModel):
    Anomaly: str


app = FastAPI(title="Anomaly Detection")


@app.get("/")
def home():
    return "Anomaly Detection Model"


@app.post("/predict", response_model=PredictionOut)
def predict(data: Input):
    pred = model_predict(dict(data))
    return {"Anomaly": pred}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
