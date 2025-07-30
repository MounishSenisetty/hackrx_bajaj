"""
Ultra-minimal API for testing Vercel deployment with new Gemini API key
"""

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from datetime import datetime

app = FastAPI(title="Test API", version="1.0.0")

class RunRequest(BaseModel):
    documents: str = Field(..., description="URL to document")
    questions: List[str] = Field(..., description="Questions")

class RunResponse(BaseModel):
    answers: List[str] = Field(..., description="Answers")
    processing_stats: Dict[str, Any] = Field(..., description="Stats")

@app.get("/")
async def root():
    return {"message": "API is working", "status": "ok", "timestamp": datetime.now().isoformat()}

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/hackrx/run", response_model=RunResponse)
async def run_submissions(request_body: RunRequest, request: Request):
    # Authentication
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication required")
    
    provided_token = auth_header.split(" ")[1]
    if provided_token != "ca6914a6c8df9d1ce075149c3ab9f060e666c75940576e37a98b3cf0e9092c72":
        raise HTTPException(status_code=403, detail="Invalid Bearer token")
    
    # Simple response indicating the new API key is configured
    answers = [f"Test response with new API key for: {q}" for q in request_body.questions]
    
    return RunResponse(
        answers=answers,
        processing_stats={
            "processing_time_seconds": 0.1,
            "questions_processed": len(request_body.questions),
            "timestamp": datetime.now().isoformat(),
            "version": "ultra-minimal",
            "gemini_api_key_updated": "AIzaSyCtvejhRMKy3NJlhu42j0LgE2td8kuUQ5o",
            "status": "successfully_deployed_with_new_key"
        }
    )

# For Vercel
app = app
