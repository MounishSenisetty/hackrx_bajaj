"""
Emergency fallback API that completely bypasses ML dependencies
to avoid 403 Forbidden errors from Google Gemini API
"""

import json
import asyncio
import os
import io
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import re

# Configure logging for Vercel
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import httpx
    from fastapi import FastAPI, Request, HTTPException
    from pydantic import BaseModel, Field
    HTTP_CLIENT_AVAILABLE = True
except ImportError as e:
    logger.error(f"Core dependencies missing: {e}")
    HTTP_CLIENT_AVAILABLE = False
    raise

# Initialize FastAPI app
app = FastAPI(
    title="LLM-Powered Intelligent Query-Retrieval System",
    description="Advanced API for processing natural language queries against documents",
    version="1.0.0"
)

# Configuration for serverless environment
class Config:
    BEARER_TOKEN = "ca6914a6c8df9d1ce075149c3ab9f060e666c75940576e37a98b3cf0e9092c72"
    REQUEST_TIMEOUT = 30.0  
    MAX_DOCUMENT_SIZE = 10 * 1024 * 1024  # 10MB limit for serverless

# --- Request/Response Models ---
class RunRequest(BaseModel):
    documents: str = Field(..., description="URL to the PDF Blob or other document type.")
    questions: List[str] = Field(..., description="List of natural language questions to ask about the documents.")

class AnswerResponse(BaseModel):
    answer: str = Field(..., description="The answer to the question")
    confidence_score: float = Field(..., description="Confidence score of the answer")
    relevant_chunks: List[Dict[str, Any]] = Field(..., description="Relevant document chunks used for the answer")
    reasoning: str = Field(..., description="Explanation of how the answer was derived")

class RunResponse(BaseModel):
    answers: List[str] = Field(..., description="List of answers corresponding to the input questions.")
    detailed_responses: Optional[List[AnswerResponse]] = Field(None, description="Detailed responses with explainability")
    processing_stats: Dict[str, Any] = Field(..., description="Processing statistics and metadata")

# --- Simple Document Processing ---
async def fetch_and_process_document_simple(url: str) -> str:
    """Fetch and process document without ML dependencies."""
    try:
        async with httpx.AsyncClient(timeout=Config.REQUEST_TIMEOUT) as client:
            response = await client.get(url)
            response.raise_for_status()
            
            content = response.content
            if len(content) > Config.MAX_DOCUMENT_SIZE:
                raise HTTPException(400, "Document exceeds size limit")
            
            # Simple text extraction
            try:
                return content.decode('utf-8', errors='ignore')
            except:
                return "Could not decode document content"
                    
    except Exception as e:
        logger.error(f"Document processing error: {e}")
        raise HTTPException(500, f"Document processing failed: {str(e)}")

# --- Simple Text Processing ---
def chunk_text_simple(text: str) -> List[Dict[str, Any]]:
    """Simple text chunking without ML dependencies."""
    chunk_size = 800
    chunks = []
    words = text.split()
    
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i + chunk_size]
        chunk_text = ' '.join(chunk_words)
        
        chunks.append({
            'text': chunk_text,
            'score': 0.5,  # Default score
            'metadata': {'chunk_id': len(chunks)}
        })
    
    return chunks

# --- Simple Search ---
def simple_keyword_search(query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Simple keyword search without ML dependencies."""
    query_words = set(re.findall(r'\w+', query.lower()))
    scored_chunks = []
    
    for chunk in chunks:
        chunk_words = set(re.findall(r'\w+', chunk['text'].lower()))
        overlap = len(query_words.intersection(chunk_words))
        
        if overlap > 0:
            score = overlap / len(query_words)
            chunk_copy = chunk.copy()
            chunk_copy['score'] = score
            scored_chunks.append(chunk_copy)
    
    # Sort by score and return top 3
    scored_chunks.sort(key=lambda x: x['score'], reverse=True)
    return scored_chunks[:3]

# --- Simple Answer Generation ---
def generate_simple_answer(question: str, relevant_chunks: List[Dict[str, Any]]) -> AnswerResponse:
    """Generate simple answers without external API calls."""
    if not relevant_chunks:
        return AnswerResponse(
            answer="Information not available in the provided document.",
            confidence_score=0.0,
            relevant_chunks=[],
            reasoning="No relevant content found for the question."
        )
    
    # Get the best chunk
    best_chunk = relevant_chunks[0]
    context = best_chunk['text']
    
    # Simple answer extraction
    question_keywords = set(re.findall(r'\w+', question.lower()))
    sentences = re.split(r'[.!?]+', context)
    
    best_sentence = ""
    best_score = 0
    
    for sentence in sentences:
        if len(sentence.strip()) < 10:
            continue
            
        sentence_words = set(re.findall(r'\w+', sentence.lower()))
        overlap = len(question_keywords.intersection(sentence_words))
        
        if overlap > best_score:
            best_score = overlap
            best_sentence = sentence.strip()
    
    if best_sentence and best_score > 0:
        answer = best_sentence
        if not answer.endswith('.'):
            answer += '.'
    else:
        # Fallback to first part of the chunk
        answer = context[:400].strip()
        if len(context) > 400:
            answer += "..."
    
    confidence = min(best_chunk.get('score', 0.5), 1.0)
    
    return AnswerResponse(
        answer=answer,
        confidence_score=confidence,
        relevant_chunks=relevant_chunks,
        reasoning=f"Answer extracted using simple text analysis from {len(relevant_chunks)} document sections. No external APIs used."
    )

# --- API Endpoints ---
@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "LLM-Powered Query-Retrieval System (Simplified)", "status": "operational"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "simplified-no-ml",
        "ml_features": False
    }

@app.post("/hackrx/run", response_model=RunResponse)
async def run_submissions(request_body: RunRequest, request: Request):
    """Main API endpoint - simplified version without ML dependencies."""
    import time
    start_time = time.time()
    
    # Authentication
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication required")
    
    provided_token = auth_header.split(" ")[1]
    if provided_token != Config.BEARER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid Bearer token")
    
    try:
        # Process document
        logger.info(f"Processing document: {request_body.documents[:100]}...")
        document_text = await fetch_and_process_document_simple(request_body.documents)
        
        if not document_text or len(document_text.strip()) < 10:
            raise HTTPException(400, "Could not extract meaningful content from document")
        
        # Simple text chunking
        chunks = chunk_text_simple(document_text)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Process questions
        answers = []
        detailed_responses = []
        
        for question in request_body.questions:
            # Simple keyword search
            relevant_chunks = simple_keyword_search(question, chunks)
            
            # Generate simple answer
            detailed_response = generate_simple_answer(question, relevant_chunks)
            
            answers.append(detailed_response.answer)
            detailed_responses.append(detailed_response)
        
        # Statistics
        processing_time = time.time() - start_time
        stats = {
            "processing_time_seconds": round(processing_time, 2),
            "document_length": len(document_text),
            "total_chunks": len(chunks),
            "questions_processed": len(request_body.questions),
            "environment": "vercel_simplified",
            "ml_features": False,
            "search_method": "keyword_based",
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Processing completed in {processing_time:.2f}s")
        
        return RunResponse(
            answers=answers,
            detailed_responses=detailed_responses,
            processing_stats=stats
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing error: {e}")
        raise HTTPException(500, f"Processing failed: {str(e)}")

# Vercel handler
handler = app
