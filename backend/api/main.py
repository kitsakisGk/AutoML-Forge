"""
FastAPI Main Application
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes import health, upload, profile, clean, train, cv_train

app = FastAPI(
    title="AutoML Pipeline Builder API",
    description="Automated ML pipeline creation and model training API with Computer Vision support",
    version="0.2.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(upload.router, prefix="/api", tags=["upload"])
app.include_router(profile.router, prefix="/api", tags=["profile"])
app.include_router(clean.router, prefix="/api", tags=["clean"])
app.include_router(train.router, prefix="/api", tags=["train"])
app.include_router(cv_train.router, prefix="/api", tags=["computer-vision"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AutoML Pipeline Builder API with Computer Vision",
        "version": "0.2.0",
        "features": ["tabular-ml", "computer-vision"],
        "docs": "/api/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
