"""Main script: it includes our API initialization and endpoints."""

import pickle
from datetime import datetime
from functools import wraps
from http import HTTPStatus
from typing import List

from fastapi import FastAPI, HTTPException, Request
from src.models.predict_model import SentiBites

model = None

# Define application
app = FastAPI(
    title="SentiBites",
    description="This API lets you make predictions on sentiment analysis of Amazon food reviews.",
    version="0.1",
)


def construct_response(f):
    """Construct a JSON response for an endpoint's results."""

    @wraps(f)
    def wrap(request: Request, *args, **kwargs):
        results = f(request, *args, **kwargs)

        # Construct response
        response = {
            "message": results["message"],
            "method": request.method,
            "status-code": results["status-code"],
            "timestamp": datetime.now().isoformat(),
            "url": request.url._url,
        }

        # Add data
        if "data" in results:
            response["data"] = results["data"]

        return response

    return wrap


@app.on_event("startup")
def _load_model():
    """Loads the model"""

    global model
    model = SentiBites("models/SentiBites/")

@app.get("/", tags=["General"])  # path operation decorator
@construct_response
def _index(request: Request):
    """Root endpoint."""

    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"message": "Welcome to SentiBites! Please, read the `/docs`!"},
    }
    return response

@app.post("/models/", tags=["Prediction"])
@construct_response
def _predict(request: Request, payload: str):
    """Performs sentiment analysis based on the food review."""

    if model:
        prediction = model.predict(payload)

        response = {
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK,
            "data": {
                "model-type": "RoBERTaSB",
                "payload": payload,
                "prediction": prediction,
            },
        }
    else:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST, detail="Model not found"
        )
    return response
