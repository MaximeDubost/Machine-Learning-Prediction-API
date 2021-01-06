from fastapi import FastAPI
import joblib
from pydantic import BaseModel

app = FastAPI()

class Main(BaseModel):
    latitude: float
    longitude: float
    beds: int
    bedrooms: int
    bathrooms: float
    pricing_weekly_factor: float
    pricing_monthly_factor: float

@app.post('/predict')
def index(args: Main):
    return joblib.load('./ml/model.joblib').predict([[
        args.latitude, 
        args.longitude, 
        args.beds, 
        args.bedrooms, 
        args.bathrooms, 
        args.pricing_weekly_factor, 
        args.pricing_monthly_factor
    ]])[0]

