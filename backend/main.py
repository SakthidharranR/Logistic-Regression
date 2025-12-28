from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

app = FastAPI(title="AI Calculation API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite default port and React default
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class NumberInput(BaseModel):
    number: float


@app.get("/")
def read_root():
    return {"message": "AI Calculation API is running"}


@app.post("/api/calculate")
def calculate_square(input_data: NumberInput):
    """
    Takes a number and squares it using numpy.
    """
    number = input_data.number
    squared = np.square(number)
    
    return {
        "input": number,
        "result": float(squared),
        "operation": "square"
    }

