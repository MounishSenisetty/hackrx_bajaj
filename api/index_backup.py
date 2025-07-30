"""
Vercel-compatible LLM-Powered Intelligent Query-Retrieval System
Optimized for serverless deployment
"""

import json
import asyncio
import os
import io
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

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

# Import simple answer generator to avoid external API issues
try:
    from simple_answer import SimpleAnswerGenerator
    SIMPLE_ANSWER_AVAILABLE = True
except ImportError:
    SIMPLE_ANSWER_AVAILABLE = False
    logger.warning("Simple answer generator not available")

# Initialize FastAPI app
app = FastAPI(
    title="LLM-Powered Intelligent Query-Retrieval System",
    description="Advanced API for processing natural language queries against documents",
    version="1.0.0"
)

# Configuration for serverless environment
class Config:
    BEARER_TOKEN = "ca6914a6c8df9d1ce075149c3ab9f060e666c75940576e37a98b3cf0e9092c72"
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBBi1RQgXXxh4CvbByxAdTB9yhZqxIqyBQ")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    MAX_CHUNK_SIZE = 800  # Reduced for serverless
    CHUNK_OVERLAP = 150
    MAX_RELEVANT_CHUNKS = 3
    REQUEST_TIMEOUT = 30.0  # Reduced for serverless
    MAX_DOCUMENT_SIZE = 10 * 1024 * 1024  # 10MB limit for serverless

# Global state for serverless
embedding_model = None
vector_capabilities = False

# --- Request/Response Models ---
class RunRequest(BaseModel):
    documents: str = Field(..., description="URL to the PDF Blob or other document type.")
    questions: List[str] = Field(..., description="List of natural language questions to ask about the documents.")

class AnswerResponse(BaseModel):
    answer: str = Field(..., description="The answer to the question")
    confidence_score: float = Field(..., description="Confidence score of the answer")
    relevant_chunks: List[Dict[str, Any]] = Field(default=[], description="Relevant document chunks")
    reasoning: str = Field(..., description="Explanation of how the answer was derived")

class RunResponse(BaseModel):
    answers: List[str] = Field(..., description="List of answers corresponding to input questions.")
    detailed_responses: Optional[List[AnswerResponse]] = Field(None, description="Detailed responses")
    processing_stats: Dict[str, Any] = Field(..., description="Processing statistics")

# --- Lazy initialization for serverless ---
async def init_ml_features():
    """Initialize ML features on demand for serverless."""
    global embedding_model, vector_capabilities
    
    if embedding_model is not None:
        return vector_capabilities
    
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        import faiss
        
        logger.info("Loading embedding model for serverless...")
        # Use a smaller, faster model for serverless
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        vector_capabilities = True
        logger.info("ML features loaded successfully")
        return True
        
    except Exception as e:
        logger.warning(f"ML features unavailable: {e}")
        vector_capabilities = False
        return False

# --- Document Processing ---
async def fetch_and_process_document(url: str) -> str:
    """Fetch and process document with size limits for serverless."""
    try:
        async with httpx.AsyncClient(timeout=Config.REQUEST_TIMEOUT) as client:
            # Get headers first to check size
            head_response = await client.head(url)
            content_length = head_response.headers.get('content-length')
            
            if content_length and int(content_length) > Config.MAX_DOCUMENT_SIZE:
                raise HTTPException(400, f"Document too large: {content_length} bytes")
            
            response = await client.get(url)
            response.raise_for_status()
            
            content = response.content
            if len(content) > Config.MAX_DOCUMENT_SIZE:
                raise HTTPException(400, "Document exceeds size limit")
            
            # Process based on content type
            content_type = response.headers.get('content-type', '').lower()
            
            if 'pdf' in content_type or url.lower().endswith('.pdf'):
                return await extract_pdf_text_serverless(content)
            else:
                try:
                    return content.decode('utf-8', errors='ignore')
                except:
                    return "Could not decode document content"
                    
    except Exception as e:
        logger.error(f"Document processing error: {e}")
        raise HTTPException(500, f"Document processing failed: {str(e)}")

async def extract_pdf_text_serverless(content: bytes) -> str:
    """Extract PDF text optimized for serverless."""
    try:
        import PyPDF2
        
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
        text_parts = []
        
        # Limit pages for serverless performance
        max_pages = min(len(pdf_reader.pages), 50)
        
        for i in range(max_pages):
            page_text = pdf_reader.pages[i].extract_text()
            if page_text.strip():
                text_parts.append(page_text)
        
        return "\n".join(text_parts)
        
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        return f"PDF processing error: {str(e)}"

# --- Text Processing ---
def chunk_text_optimized(text: str) -> List[Dict[str, Any]]:
    """Optimized text chunking for serverless."""
    sentences = text.replace('\n', ' ').split('. ')
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence.split())
        
        if current_length + sentence_length <= Config.MAX_CHUNK_SIZE:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            if current_chunk:
                chunk_text = '. '.join(current_chunk) + '.'
                chunks.append({
                    'text': chunk_text,
                    'chunk_id': len(chunks),
                    'length': current_length,
                    'score': 1.0
                })
            
            current_chunk = [sentence]
            current_length = sentence_length
    
    # Add final chunk
    if current_chunk:
        chunk_text = '. '.join(current_chunk) + '.'
        chunks.append({
            'text': chunk_text,
            'chunk_id': len(chunks),
            'length': current_length,
            'score': 1.0
        })
    
    return chunks

# --- Search Functions ---
async def semantic_search_serverless(query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Serverless-optimized semantic search."""
    global embedding_model
    
    has_ml = await init_ml_features()
    
    if not has_ml or not embedding_model:
        return keyword_search_enhanced(query, chunks)
    
    try:
        import numpy as np
        import faiss
        
        if len(chunks) == 0:
            return []
        
        # Generate embeddings efficiently
        chunk_texts = [chunk['text'] for chunk in chunks]
        chunk_embeddings = embedding_model.encode(chunk_texts, show_progress_bar=False)
        query_embedding = embedding_model.encode([query], show_progress_bar=False)
        
        # Use simple cosine similarity for small datasets
        if len(chunks) <= 20:
            similarities = np.dot(chunk_embeddings, query_embedding.T).flatten()
            top_indices = np.argsort(similarities)[::-1][:Config.MAX_RELEVANT_CHUNKS]
            
            results = []
            for i, idx in enumerate(top_indices):
                chunk = chunks[idx].copy()
                chunk['score'] = float(similarities[idx])
                chunk['rank'] = i + 1
                results.append(chunk)
            
            return results
        else:
            # Use FAISS for larger datasets
            dimension = chunk_embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            
            faiss.normalize_L2(chunk_embeddings)
            faiss.normalize_L2(query_embedding)
            index.add(chunk_embeddings.astype('float32'))
            
            scores, indices = index.search(query_embedding.astype('float32'), Config.MAX_RELEVANT_CHUNKS)
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx != -1:
                    chunk = chunks[idx].copy()
                    chunk['score'] = float(score)
                    chunk['rank'] = i + 1
                    results.append(chunk)
            
            return results
            
    except Exception as e:
        logger.error(f"Semantic search error: {e}")
        return keyword_search_enhanced(query, chunks)

def keyword_search_enhanced(query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Enhanced keyword search with better scoring."""
    query_words = set(query.lower().split())
    scored_chunks = []
    
    for chunk in chunks:
        chunk_words = set(chunk['text'].lower().split())
        
        # Calculate different types of matches
        exact_matches = len(query_words.intersection(chunk_words))
        partial_matches = sum(1 for qw in query_words for cw in chunk_words if qw in cw or cw in qw)
        
        # Weighted score
        score = (exact_matches * 2 + partial_matches * 0.5) / len(query_words)
        
        if score > 0:
            chunk_copy = chunk.copy()
            chunk_copy['score'] = score
            scored_chunks.append(chunk_copy)
    
    # Sort and return top chunks
    scored_chunks.sort(key=lambda x: x['score'], reverse=True)
    return scored_chunks[:Config.MAX_RELEVANT_CHUNKS]

# --- Answer Generation ---
async def generate_precise_answer(question: str, relevant_chunks: List[Dict[str, Any]]) -> AnswerResponse:
    """Generate precise answers from relevant chunks without external API calls."""
    try:
        # Use simple answer generator to avoid 403 errors
        if SIMPLE_ANSWER_AVAILABLE:
            result = SimpleAnswerGenerator.generate_answer(question, relevant_chunks)
            return AnswerResponse(**result)
    except Exception as e:
        logger.warning(f"Simple answer generator failed: {e}")
    
    # Fallback implementation
    if not relevant_chunks:
        return AnswerResponse(
            answer="Information not available in the provided document.",
            confidence_score=0.0,
            relevant_chunks=[],
            reasoning="No relevant content found for the question."
        )
    
    # Extract the most relevant information
    best_chunk = relevant_chunks[0]
    context = best_chunk['text']
    
    # Try to find precise answers within the context
    question_keywords = set(question.lower().split())
    sentences = context.split('.')
    
    best_sentence = ""
    best_score = 0
    
    for sentence in sentences:
        sentence_words = set(sentence.lower().split())
        overlap = len(question_keywords.intersection(sentence_words))
        
        if overlap > best_score and len(sentence.strip()) > 10:
            best_score = overlap
            best_sentence = sentence.strip()
    
    # Use best sentence or fallback to chunk summary
    if best_sentence and best_score > 0:
        answer = best_sentence
        if not answer.endswith('.'):
            answer += '.'
    else:
        # Extract key information from the chunk
        answer = context[:400].strip()
        if len(context) > 400:
            answer += "..."
    
    confidence = min(best_chunk['score'], 1.0)
    
    # NO EXTERNAL API CALLS TO AVOID 403 ERRORS
    return AnswerResponse(
        answer=answer,
        confidence_score=confidence,
        relevant_chunks=relevant_chunks,
        reasoning=f"Answer extracted from {len(relevant_chunks)} relevant document sections using text analysis (no external API calls to avoid 403 errors)."
    )

# --- API Endpoints ---
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "LLM-Powered Intelligent Query-Retrieval System",
        "status": "active",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "system_info": "/system/info",
            "main_api": "/hackrx/run"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "environment": "serverless",
        "version": "1.0.0"
    }

@app.get("/system/info")
async def system_info():
    """System information."""
    has_ml = await init_ml_features()
    return {
        "system_name": "LLM-Powered Intelligent Query-Retrieval System",
        "environment": "Vercel Serverless",
        "capabilities": [
            "PDF document processing",
            "Text chunking with optimization",
            "Semantic search" if has_ml else "Keyword search",
            "Precise answer extraction",
            "Real-time processing"
        ],
        "ml_features_available": has_ml,
        "max_document_size": f"{Config.MAX_DOCUMENT_SIZE // (1024*1024)}MB",
        "max_chunk_size": Config.MAX_CHUNK_SIZE
    }

@app.post("/hackrx/run", response_model=RunResponse)
async def run_submissions(request_body: RunRequest, request: Request):
    """Main API endpoint for processing documents and answering questions."""
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
        document_text = await fetch_and_process_document(request_body.documents)
        
        if not document_text or len(document_text.strip()) < 10:
            raise HTTPException(400, "Could not extract meaningful content from document")
        
        # Chunk text
        chunks = chunk_text_optimized(document_text)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Process questions
        answers = []
        detailed_responses = []
        
        for question in request_body.questions:
            # Search for relevant content
            relevant_chunks = await semantic_search_serverless(question, chunks)
            
            # Generate precise answer
            detailed_response = await generate_precise_answer(question, relevant_chunks)
            
            answers.append(detailed_response.answer)
            detailed_responses.append(detailed_response)
        
        # Statistics
        processing_time = time.time() - start_time
        stats = {
            "processing_time_seconds": round(processing_time, 2),
            "document_length": len(document_text),
            "total_chunks": len(chunks),
            "questions_processed": len(request_body.questions),
            "environment": "vercel_serverless",
            "ml_features": vector_capabilities,
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
