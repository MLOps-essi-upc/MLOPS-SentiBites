from fastapi.testclient import TestClient
import os
import sys

# Get the parent directory
parent_dir = os.path.dirname(os.path.realpath(__file__))

# Add the parent directory to sys.path
sys.path.append(parent_dir)

from src.app.api import app
from http import HTTPStatus

def test_read_main():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        response_body = response.json()
        assert response_body["message"] == HTTPStatus.OK.phrase
        assert response_body["data"]["message"] == "Welcome to SentiBites! Please, read the `/docs`!"

def test_read_prediction():
    with TestClient(app) as client:
        response = client.post("/models", json = {'msg':"This is a test."})
        assert response.status_code == 200
        response_body = response.json()
        assert response_body["message"] == HTTPStatus.OK.phrase
        assert response_body["status-code"] == HTTPStatus.OK
        assert response_body["data"]["model-type"] == "RoBERTaSB"
        assert response_body["data"]["payload"] == "This is a test."

def test_positive_prediction():
    with TestClient(app) as client:
        response = client.post("/models/", json={"msg": "This food is really good."})
        assert response.status_code == 200
        response_body = response.json()
        assert response_body["message"] == HTTPStatus.OK.phrase
        assert response_body["status-code"] == HTTPStatus.OK
        assert response_body["data"]["model-type"] == "RoBERTaSB"
        assert response_body["data"]["payload"] == "This food is really good."
        assert response_body["data"]["prediction"] == "positive"

def test_negative_prediction():
    with TestClient(app) as client:
        response = client.post("/models/", json={"msg": "Never buying this again."})
        assert response.status_code == 200
        response_body = response.json()
        assert response_body["message"] == HTTPStatus.OK.phrase
        assert response_body["status-code"] == HTTPStatus.OK
        assert response_body["data"]["model-type"] == "RoBERTaSB"
        assert response_body["data"]["payload"] == "Never buying this again."
        assert response_body["data"]["prediction"] == "negative"

def test_bad_url():
    with TestClient(app) as client:
        response = client.post("/mode", json={"msg": "Never buying this again."})
        assert response.status_code == 404

def test_bad_request():
    with TestClient(app) as client:
        response = client.post("/models/", json={"false": "Never buying this again."})
        assert response.status_code == 422