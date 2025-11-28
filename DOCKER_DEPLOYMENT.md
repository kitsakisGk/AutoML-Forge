# 🐳 Docker Deployment Guide

This guide shows how to deploy AutoML Forge using Docker containers.

## 📋 Prerequisites

- Docker Desktop installed: https://www.docker.com/products/docker-desktop
- Docker Compose (included with Docker Desktop)
- 8GB RAM minimum
- 20GB disk space

## 🚀 Quick Start

### Option 1: Docker Compose (Recommended)

Deploy both backend and frontend with a single command:

```bash
# Build and start both services
docker-compose up --build

# Run in background (detached mode)
docker-compose up -d --build
```

Access the application:
- **Frontend**: http://localhost:8501
- **Backend API**: http://localhost:8000/api/docs

Stop the services:
```bash
docker-compose down
```

### Option 2: Individual Containers

**Build Backend:**
```bash
docker build -f Dockerfile.backend -t automl-forge-backend .
docker run -p 8000:8000 -v $(pwd)/uploads:/app/uploads automl-forge-backend
```

**Build Frontend:**
```bash
docker build -f Dockerfile.frontend -t automl-forge-frontend .
docker run -p 8501:8501 automl-forge-frontend
```

## 📦 What's Included

### Backend Container
- FastAPI server on port 8000
- All ML dependencies (scikit-learn, XGBoost, LightGBM, PyTorch)
- MLflow tracking
- Persistent volumes for uploads and experiments

### Frontend Container
- Streamlit UI on port 8501
- Connected to backend via Docker network
- Bilingual support (EN/DE)

## 🔧 Configuration

### Environment Variables

Create a `.env` file:
```env
# Backend
BACKEND_PORT=8000
UPLOAD_DIR=/app/uploads
MLFLOW_TRACKING_URI=file:./mlruns

# Frontend
FRONTEND_PORT=8501
API_BASE_URL=http://backend:8000/api
```

Use with docker-compose:
```bash
docker-compose --env-file .env up
```

### GPU Support (Optional)

For faster CV training, enable GPU in docker-compose.yml:

```yaml
services:
  backend:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

Requires:
- NVIDIA GPU
- nvidia-docker installed

## 📊 Volume Persistence

Data persists across container restarts:

- **uploads/**: Uploaded datasets and trained models
- **mlruns/**: MLflow experiment tracking

To backup:
```bash
docker-compose down
tar -czf automl-backup.tar.gz uploads/ mlruns/
```

To restore:
```bash
tar -xzf automl-backup.tar.gz
docker-compose up -d
```

## 🐛 Troubleshooting

### Port already in use
```bash
# Change ports in docker-compose.yml
ports:
  - "8001:8000"  # Backend
  - "8502:8501"  # Frontend
```

### Out of memory
```bash
# Increase Docker memory limit in Docker Desktop settings
# Minimum 8GB recommended for CV training
```

### Permission issues
```bash
# Fix volume permissions
sudo chown -R $(id -u):$(id -g) uploads/ mlruns/
```

## 🌐 Production Deployment

### Deploy to Cloud

**AWS ECS:**
```bash
# Push images to ECR
aws ecr create-repository --repository-name automl-forge-backend
docker tag automl-forge-backend:latest <aws-account-id>.dkr.ecr.us-east-1.amazonaws.com/automl-forge-backend:latest
docker push <aws-account-id>.dkr.ecr.us-east-1.amazonaws.com/automl-forge-backend:latest
```

**Google Cloud Run:**
```bash
# Build and push
gcloud builds submit --tag gcr.io/<project-id>/automl-forge-backend
gcloud run deploy --image gcr.io/<project-id>/automl-forge-backend --platform managed
```

**DigitalOcean:**
```bash
# Use docker-compose.yml directly
doctl apps create --spec docker-compose.yml
```

## 📝 Docker Best Practices

1. **Multi-stage builds**: Reduce image size
2. **Layer caching**: Order COPY commands efficiently
3. **Health checks**: Add health endpoints
4. **Secrets**: Use Docker secrets for sensitive data
5. **Logging**: Configure logging drivers

## 🔐 Security

```yaml
# Add health checks
services:
  backend:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## 📖 Additional Resources

- Docker Documentation: https://docs.docker.com
- Docker Compose: https://docs.docker.com/compose
- Best Practices: https://docs.docker.com/develop/dev-best-practices

---

**Author:** Kitsakis Giorgos
**GitHub:** [kitsakisGk/AutoML-Forge](https://github.com/kitsakisGk/AutoML-Forge)
