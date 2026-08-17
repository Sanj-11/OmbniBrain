from fastapi import FastAPI
from pydantic import BaseModel

# Define response models for better type hinting and documentation
class RootResponse(BaseModel):
    message: str
    status: str

class HealthResponse(BaseModel):
    status: str

# Create the FastAPI application instance
app = FastAPI(
    title="OmniBrain API",
    version="1.0.0",
)

@app.get("/", response_model=RootResponse)
async def read_root():
    return {"message": "OmniBrain API is running", "status": "success"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return {"status": "healthy"}