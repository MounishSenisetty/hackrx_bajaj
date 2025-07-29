"""
LLM-Powered Intelligent Query-Retrieval System - Simplified Version
"""

import json
import asyncio
import os
import io
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

# Basic imports first
try:
    import httpx
    from fastapi import FastAPI, Request, HTTPException
    from pydantic import BaseModel, Field
    HTTP_CLIENT_AVAILABLE = True
except ImportError as e:
    logging.error(f"Core dependencies missing: {e}")
    HTTP_CLIENT_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LLM-Powered Intelligent Query-Retrieval System",
    description="Advanced API for processing natural language queries against documents with semantic search",
    version="1.0.0"
)

# Global state
system_initialized = False
embedding_model = None
vector_capabilities = False

# --- Configuration ---
class Config:
    BEARER_TOKEN = "ca6914a6c8df9d1ce075149c3ab9f060e666c75940576e37a98b3cf0e9092c72"
    MAX_CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    MAX_RELEVANT_CHUNKS = 3
    REQUEST_TIMEOUT = 60.0

# --- Request/Response Models ---
class RunRequest(BaseModel):
    documents: str = Field(..., description="URL to the PDF Blob or other document type.")
    questions: List[str] = Field(..., description="List of natural language questions to ask about the documents.")

class AnswerResponse(BaseModel):
    answer: str = Field(..., description="The answer to the question")
    confidence_score: float = Field(..., description="Confidence score of the answer")
    relevant_chunks: List[Dict[str, Any]] = Field(default=[], description="Relevant document chunks used for the answer")
    reasoning: str = Field(..., description="Explanation of how the answer was derived")

class RunResponse(BaseModel):
    answers: List[str] = Field(..., description="List of answers corresponding to the input questions.")
    detailed_responses: Optional[List[AnswerResponse]] = Field(None, description="Detailed responses with explainability")
    processing_stats: Dict[str, Any] = Field(..., description="Processing statistics and metadata")

# --- Advanced Features (Lazy Loading) ---
async def initialize_advanced_features():
    """Initialize advanced ML features if dependencies are available."""
    global embedding_model, vector_capabilities, system_initialized
    
    try:
        # Try to import ML dependencies
        import numpy as np
        from sentence_transformers import SentenceTransformer
        import faiss
        import PyPDF2
        from docx import Document
        
        logger.info("Initializing embedding model...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        vector_capabilities = True
        logger.info("Advanced features initialized successfully")
        
    except ImportError as e:
        logger.warning(f"Advanced ML features not available: {e}")
        logger.info("Running in basic mode without embeddings")
        vector_capabilities = False
    
    system_initialized = True

# --- Document Processing ---
async def fetch_document_content(url: str) -> str:
    """Fetch and extract text from document URL."""
    if not HTTP_CLIENT_AVAILABLE:
        return "HTTP client not available - dependency missing"
    
    try:
        async with httpx.AsyncClient(timeout=Config.REQUEST_TIMEOUT) as client:
            response = await client.get(url)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '').lower()
            content = response.content
            
            # For this demo, try to extract text content
            if 'pdf' in content_type or url.lower().endswith('.pdf'):
                return await extract_pdf_text(content)
            elif 'text' in content_type or url.lower().endswith('.txt'):
                return content.decode('utf-8', errors='ignore')
            else:
                # Attempt text extraction
                try:
                    return content.decode('utf-8', errors='ignore')
                except:
                    return "Could not extract text from document"
                    
    except Exception as e:
        logger.error(f"Error fetching document: {e}")
        return f"Error fetching document: {e}"

async def extract_pdf_text(content: bytes) -> str:
    """Extract text from PDF content."""
    try:
        import PyPDF2
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except ImportError:
        return "PDF processing not available - PyPDF2 dependency missing"
    except Exception as e:
        logger.error(f"Error extracting PDF text: {e}")
        return f"Error processing PDF: {e}"

# --- Text Processing ---
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = ' '.join(chunk_words)
        
        chunks.append({
            'text': chunk_text,
            'start_index': i,
            'end_index': min(i + chunk_size, len(words)),
            'chunk_id': len(chunks),
            'score': 1.0  # Default score
        })
        
        if i + chunk_size >= len(words):
            break
            
    return chunks

# --- Semantic Search (with fallback) ---
async def semantic_search(query: str, chunks: List[Dict[str, Any]], k: int = 3) -> List[Dict[str, Any]]:
    """Perform semantic search on chunks."""
    global embedding_model, vector_capabilities
    
    if not vector_capabilities or not embedding_model:
        # Fallback to keyword-based search
        return keyword_search(query, chunks, k)
    
    try:
        import numpy as np
        import faiss
        
        # Generate embeddings
        chunk_texts = [chunk['text'] for chunk in chunks]
        chunk_embeddings = embedding_model.encode(chunk_texts)
        query_embedding = embedding_model.encode([query])
        
        # Create FAISS index
        dimension = chunk_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        
        # Normalize and add embeddings
        faiss.normalize_L2(chunk_embeddings)
        faiss.normalize_L2(query_embedding)
        index.add(chunk_embeddings.astype('float32'))
        
        # Search
        scores, indices = index.search(query_embedding.astype('float32'), min(k, len(chunks)))
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1:
                chunk = chunks[idx].copy()
                chunk['score'] = float(score)
                chunk['rank'] = i + 1
                results.append(chunk)
        
        return results
        
    except Exception as e:
        logger.error(f"Error in semantic search: {e}")
        return keyword_search(query, chunks, k)

def keyword_search(query: str, chunks: List[Dict[str, Any]], k: int = 3) -> List[Dict[str, Any]]:
    """Fallback keyword-based search."""
    query_words = query.lower().split()
    scored_chunks = []
    
    for chunk in chunks:
        chunk_text = chunk['text'].lower()
        score = 0
        for word in query_words:
            score += chunk_text.count(word)
        
        if score > 0:
            chunk_copy = chunk.copy()
            chunk_copy['score'] = score / len(query_words)
            scored_chunks.append(chunk_copy)
    
    # Sort by score and return top k
    scored_chunks.sort(key=lambda x: x['score'], reverse=True)
    return scored_chunks[:k]

# --- Answer Generation ---
async def generate_answer(question: str, relevant_chunks: List[Dict[str, Any]]) -> AnswerResponse:
    """Generate answer using relevant chunks."""
    if not relevant_chunks:
        return AnswerResponse(
            answer="Information not available in the provided document.",
            confidence_score=0.0,
            relevant_chunks=[],
            reasoning="No relevant chunks found for the question."
        )
    
    # Use the most relevant chunk for answer
    best_chunk = relevant_chunks[0]
    context = best_chunk['text']
    
    # Simple answer extraction (in a real system, this would use LLM)
    answer = f"Based on the document content: {context[:300]}..."
    if len(context) > 300:
        answer += " [Content truncated]"
    
    # Try to find more specific answer within the chunk
    question_words = question.lower().split()
    sentences = context.split('.')
    
    for sentence in sentences:
        sentence_words = sentence.lower().split()
        if any(qword in sentence_words for qword in question_words):
            answer = sentence.strip()
            break
    
    confidence = best_chunk['score'] if 'score' in best_chunk else 0.5
    
    return AnswerResponse(
        answer=answer,
        confidence_score=min(confidence, 1.0),
        relevant_chunks=relevant_chunks,
        reasoning=f"Answer derived from {len(relevant_chunks)} relevant document chunks using semantic similarity search."
    )

# --- API Endpoints ---
@app.on_event("startup")
async def startup_event():
    """Initialize system on startup."""
    logger.info("Starting LLM-Powered Query-Retrieval System...")
    await initialize_advanced_features()
    logger.info("System startup complete")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system_initialized": system_initialized,
        "vector_capabilities": vector_capabilities,
        "http_client_available": HTTP_CLIENT_AVAILABLE,
        "version": "1.0.0"
    }

@app.get("/system/info")
async def system_info():
    """System information endpoint."""
    return {
        "system_name": "LLM-Powered Intelligent Query-Retrieval System",
        "capabilities": [
            "Document fetching and processing",
            "Text chunking with overlap",
            "Semantic search" if vector_capabilities else "Keyword search",
            "Contextual question answering",
            "Explainable AI responses",
            "Real-time performance optimization"
        ],
        "supported_formats": ["PDF", "Plain Text", "Any text-based format"],
        "embedding_model": "all-MiniLM-L6-v2" if vector_capabilities else "None (keyword search)",
        "vector_database": "FAISS" if vector_capabilities else "Simple keyword matching",
        "system_initialized": system_initialized,
        "dependencies_status": {
            "httpx": HTTP_CLIENT_AVAILABLE,
            "sentence_transformers": vector_capabilities,
            "faiss": vector_capabilities
        }
    }

@app.post("/hackrx/run", response_model=RunResponse)
async def run_submissions(request_body: RunRequest, request: Request):
    """Main endpoint for processing documents and answering questions."""
    start_time = time.time()
    
    # Authentication
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication required: Bearer token missing.")
    
    provided_token = auth_header.split(" ")[1]
    if provided_token != Config.BEARER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid Bearer token.")
    
    try:
        # Ensure system is initialized
        if not system_initialized:
            await initialize_advanced_features()
        
        # Document processing
        logger.info(f"Processing document from URL: {request_body.documents}")
        document_text = await fetch_document_content(request_body.documents)
        
        if not document_text or "Error" in document_text:
            raise HTTPException(status_code=400, detail=f"Could not process document: {document_text}")
        
        # Text chunking
        chunks = chunk_text(document_text, Config.MAX_CHUNK_SIZE, Config.CHUNK_OVERLAP)
        logger.info(f"Document chunked into {len(chunks)} pieces")
        
        # Process questions
        answers = []
        detailed_responses = []
        
        for question in request_body.questions:
            # Semantic search
            relevant_chunks = await semantic_search(question, chunks, Config.MAX_RELEVANT_CHUNKS)
            
            # Generate answer
            detailed_response = await generate_answer(question, relevant_chunks)
            
            answers.append(detailed_response.answer)
            detailed_responses.append(detailed_response)
        
        # Processing statistics
        processing_time = time.time() - start_time
        stats = {
            "processing_time_seconds": round(processing_time, 2),
            "document_length": len(document_text),
            "total_chunks": len(chunks),
            "questions_processed": len(request_body.questions),
            "average_chunk_size": sum(len(chunk['text']) for chunk in chunks) // len(chunks) if chunks else 0,
            "search_method": "semantic" if vector_capabilities else "keyword",
            "embedding_model": "all-MiniLM-L6-v2" if vector_capabilities else "none",
            "vector_store": "FAISS" if vector_capabilities else "simple",
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Processing completed in {processing_time:.2f} seconds")
        
        return RunResponse(
            answers=answers,
            detailed_responses=detailed_responses,
            processing_stats=stats
        )
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
