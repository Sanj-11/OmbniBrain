from fastapi import FastAPI
from database import engine, Base
from config import settings

import models
import auth
import datasets
import profiling
import cleaning
import query

# Create database tables
models.Base.metadata.create_all(bind=engine)

# Ensure upload directory exists
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Create the FastAPI application instance
app = FastAPI(
    title="OmniBrain API",
    version="1.0.0",
    description="API for InsightIQ AI – Conversational BI & Predictive Analytics Platform"
)

app.include_router(auth.router)
app.include_router(datasets.router)
app.include_router(profiling.router)
app.include_router(cleaning.router)
app.include_router(query.router)

@app.get("/")
async def read_root():
    return {"message": "OmniBrain API is running", "status": "success"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}