from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import realtime, video_upload




app = FastAPI(
    title="SignLang Translator Backend",
    description="API for real-time sign language to English translation",
    version="0.1.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(realtime.router, prefix="/api/realtime")
app.include_router(video_upload.router, prefix="/api/video")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
