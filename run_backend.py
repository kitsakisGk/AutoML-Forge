"""
Quick start script for backend
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "backend.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        timeout_keep_alive=600  # 10 minutes for CV training
    )
