"""
Run API server with proper configuration for concurrent requests
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8500,
        workers=1,  # Use 1 worker since we're using async
        log_level="info",
        access_log=True
    )