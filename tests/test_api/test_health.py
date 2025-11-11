"""
Tests for health endpoint
"""
import pytest
from fastapi.testclient import TestClient
from backend.api.main import app

client = TestClient(app)


def test_health_check():
    """Test health check endpoint"""
    response = client.get("/api/health")

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert data["service"] == "automl-pipeline-builder"


def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()

    assert "message" in data
    assert "version" in data
    assert data["version"] == "0.1.0"
