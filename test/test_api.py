from fastapi.testclient import TestClient
from src.app.main import app
from pydantic import ValidationError
import unittest

client = TestClient(app)

def test_health():
    # Test the health endpoint
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

# A few tests to check the model is working as expected

def test_predict_positive():
    response = client.post("/predict", json={"text": "I love this product", "asin": "abcdefghij", "parent_asin": "abcdefghij"})
    assert response.status_code == 200
    assert response.json()["sentiment"] == "positive"

def test_predict_negative():
    response = client.post("/predict", json={"text": "I hate this product", "asin": "abcdefghij", "parent_asin": "abcdefghij"})
    assert response.status_code == 200
    assert response.json()["sentiment"] == "negative"

# Check pydantic is throwing validation errors for invalid inputs

def test_empty_text_field():
    response = client.post("/predict", json={"text": "", "asin": "abcdefghij", "parent_asin": "abcdefghij"})
    assert response.status_code == 422


def test_shorter_asin():
    response = client.post("/predict", json={"text": "I love this product", "asin": "abcdefghi", "parent_asin": "abcdefghij"})
    assert response.status_code == 422
