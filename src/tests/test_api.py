from fastapi.testclient import TestClient
from src.app.api import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to SentiBites! Please, read the `/docs`!"}