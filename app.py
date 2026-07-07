from fastapi import FastAPI
from fastapi.responses import JSONResponse
from typing import Annotated
from pydantic import BaseModel,Field
import pickle
from xgboost import XGBClassifier
import numpy as np
import pandas as pd

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # sab allow (development ke liye)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# model load karte hai
with open('model.pkl','rb') as f:
    model=pickle.load(f)

# transformation ko load karlo
with open('transformation.pkl','rb') as f:
    transform=pickle.load(f)


@app.get('/')
def home():
    return {'message':'This is the home page of my api'}

class InputCheck(BaseModel):
    amount:Annotated[float,Field(...,gt=0)]
    transaction_hour:Annotated[int,Field(...,gt=0,le=23)]
    merchant_category:str
    foreign_transaction:bool
    location_mismatch:bool
    device_trust_score:int
    velocity_last_24h:Annotated[int,Field(...,gt=0)]
    cardholder_age:Annotated[int,Field(...,gt=0)]

@app.post('/predict')
def predict(data: InputCheck):
    df = pd.DataFrame([data.dict()])
    trf_data = transform.transform(df)
    prediction = model.predict(trf_data)
    return JSONResponse(status_code=200, content={'prediction of fraud': prediction.tolist()}) 