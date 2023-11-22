from fastapi.testclient import TestClient
from src.app.api import app
from http import HTTPStatus

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to SentiBites! Please, read the `/docs`!"}

def test_read_prediction():
    response = client.post("/models/", json={"payload": "This is a test."})
    assert response.status_code == 200
    response_body = response.json()
    assert response_body["message"] == HTTPStatus.OK.phrase
    assert response_body["status-code"] == HTTPStatus.OK
    assert response_body["data"]["model-type"] == "RoBERTaSB"
    assert response_body["data"]["payload"] == "This is a test."

def test_positive_prediction():
    response = client.post("/models/", json={"payload": "This food is really good."})
    assert response.status_code == 200
    response_body = response.json()
    assert response_body["message"] == HTTPStatus.OK.phrase
    assert response_body["status-code"] == HTTPStatus.OK
    assert response_body["data"]["model-type"] == "RoBERTaSB"
    assert response_body["data"]["payload"] == "This food is really good."
    assert response_body["data"]["prediction"] == "positive"

def test_negative_prediction():
    response = client.post("/models/", json={"payload": "Never buying this again."})
    assert response.status_code == 200
    response_body = response.json()
    assert response_body["message"] == HTTPStatus.OK.phrase
    assert response_body["status-code"] == HTTPStatus.OK
    assert response_body["data"]["model-type"] == "RoBERTaSB"
    assert response_body["data"]["payload"] == "Never buying this again."
    assert response_body["data"]["prediction"] == "negative"