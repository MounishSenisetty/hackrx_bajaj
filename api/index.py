"""
Vercel-Compatible Document Query System with Vector Search
"""

import os
# Load .env automatically for local development
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import re
import io
import json
import httpx
import numpy as np
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


app = FastAPI(title="Vercel-Compatible Document Query System", version="1.0.0")

# Configuration
class Config:
    BEARER_TOKEN = os.getenv("BEARER_TOKEN")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
    CHUNK_SIZE = 800  # characters per chunk
    CHUNK_OVERLAP = 100
    MAX_CHUNKS = 20
    REQUEST_TIMEOUT = 10.0
    MAX_DOCUMENT_SIZE = 10 * 1024 * 1024
    SIMILARITY_THRESHOLD = 0.7
    MAX_RELEVANT_CHUNKS = 3





# Models
class RunRequest(BaseModel):
    documents: str = Field(..., description="URL to document")
    questions: list[str] = Field(..., description="Questions")

class RunResponse(BaseModel):
    answers: list[str] = Field(..., description="List of answers")


# Document Processing
async def fetch_document(url: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=Config.REQUEST_TIMEOUT, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            content = response.content
            if len(content) > Config.MAX_DOCUMENT_SIZE:
                raise HTTPException(400, "Document exceeds size limit")
            content_type = response.headers.get('content-type', '').lower()
            is_pdf = 'pdf' in content_type or url.lower().endswith('.pdf')
            if is_pdf and PDF_AVAILABLE:
                return extract_pdf_text(content)
            else:
                return extract_text_fallback(content)
    except Exception as e:
        raise HTTPException(500, f"Document fetch failed: {e}")

def extract_pdf_text(content: bytes) -> str:
    try:
        pdf_file = io.BytesIO(content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = "".join(page.extract_text() + "\n" for page in pdf_reader.pages if page.extract_text())
        text = re.sub(r'\s+', ' ', text).strip()
        return text if len(text) >= 50 else extract_text_fallback(content)
    except Exception as e:
        print(f"PDF extraction failed: {e}")
        return extract_text_fallback(content)

def extract_text_fallback(content: bytes) -> str:
    try:
        text = content.decode('utf-8', errors='ignore')
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text) < 100:
            text_latin1 = content.decode('latin-1', errors='ignore')
            text_latin1 = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', ' ', text_latin1)
            text_latin1 = re.sub(r'\s+', ' ', text_latin1).strip()
            if len(text_latin1) > len(text):
                text = text_latin1
        return text or "Document content could not be decoded"
    except Exception:
        return "Document content could not be decoded"

def smart_chunk_text(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(current_chunk) + len(sentence) + 1 > Config.CHUNK_SIZE:
            if current_chunk:
                chunks.append(current_chunk)
            if len(sentence) > Config.CHUNK_SIZE:
                chunks.extend([sentence[i:i+Config.CHUNK_SIZE] for i in range(0, len(sentence), Config.CHUNK_SIZE)])
                current_chunk = ""
            else:
                current_chunk = sentence
        else:
            current_chunk += (" " + sentence) if current_chunk else sentence
    if current_chunk:
        chunks.append(current_chunk)
    
    if Config.CHUNK_OVERLAP == 0 or len(chunks) <= 1:
        return chunks[:Config.MAX_CHUNKS]

    overlapped_chunks = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_chunk = chunks[i-1]
        overlap = prev_chunk[-Config.CHUNK_OVERLAP:]
        space_index = overlap.find(' ')
        if space_index != -1:
            overlap = overlap[space_index+1:]
        overlapped_chunks.append(overlap.strip() + " " + chunks[i])
    return overlapped_chunks[:Config.MAX_CHUNKS]



# Embedding and Vector Search Functions
async def get_openai_embedding(text: str) -> List[float]:
    """Get embeddings from OpenAI API"""
    if not Config.OPENAI_API_KEY:
        return None
    
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {Config.OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "input": text[:8000],  # Limit input size
        "model": "text-embedding-3-small"
    }
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                result = response.json()
                return result["data"][0]["embedding"]
    except Exception:
        pass
    return None

async def get_google_embedding(text: str) -> List[float]:
    """Get embeddings from Google API"""
    if not Config.GOOGLE_API_KEY:
        return None
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key={Config.GOOGLE_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "models/embedding-001",
        "content": {"parts": [{"text": text[:8000]}]}
    }
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                result = response.json()
                return result["embedding"]["values"]
    except Exception:
        pass
    return None

def simple_embedding(text: str) -> List[float]:
    """Fallback: Simple TF-IDF like embedding"""
    words = re.findall(r'\w+', text.lower())
    # Create a simple hash-based embedding
    embedding = [0.0] * 384  # Standard embedding size
    for i, word in enumerate(words[:50]):  # Limit to 50 words
        hash_val = hash(word) % 384
        embedding[hash_val] += 1.0 / (i + 1)  # Position weight
    
    # Normalize
    norm = sum(x*x for x in embedding) ** 0.5
    if norm > 0:
        embedding = [x/norm for x in embedding]
    return embedding

async def get_embedding(text: str) -> List[float]:
    """Get embeddings with fallback chain: OpenAI -> Google -> Simple"""
    # Try OpenAI first (fastest and most accurate)
    embedding = await get_openai_embedding(text)
    if embedding:
        return embedding
    
    # Try Google if OpenAI fails
    embedding = await get_google_embedding(text)
    if embedding:
        return embedding
    
    # Fallback to simple embedding
    return simple_embedding(text)

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    if len(a) != len(b):
        return 0.0
    
    dot_product = sum(x*y for x, y in zip(a, b))
    norm_a = sum(x*x for x in a) ** 0.5
    norm_b = sum(x*x for x in b) ** 0.5
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)





# Generative APIs with fallback
async def call_openai_api(question: str, context: str) -> str:
    """Call OpenAI GPT API"""
    if not Config.OPENAI_API_KEY:
        return None
    
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {Config.OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "Answer the question based on the provided context. Be concise and accurate."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ],
        "max_tokens": 150,
        "temperature": 0.1
    }
    
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
    except Exception:
        pass
    return None

async def call_google_api(question: str, context: str) -> str:
    """Call Google Gemini API"""
    if not Config.GOOGLE_API_KEY:
        return None
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={Config.GOOGLE_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{
            "parts": [{
                "text": f"Based on this context, answer the question concisely:\n\nContext: {context}\n\nQuestion: {question}"
            }]
        }],
        "generationConfig": {
            "maxOutputTokens": 150,
            "temperature": 0.1
        }
    }
    
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                result = response.json()
                return result["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        pass
    return None

def extract_answer_from_context(question: str, context: str) -> str:
    """Simple rule-based answer extraction as fallback"""
    question_lower = question.lower()
    context_lower = context.lower()
    
    # Look for direct answers to common question patterns
    if "what is" in question_lower or "what are" in question_lower:
        sentences = re.split(r'[.!?]+', context)
        for sentence in sentences:
            if any(word in sentence.lower() for word in question_lower.split()[2:]):
                return sentence.strip()
    
    # For other questions, return the most relevant sentence
    sentences = re.split(r'[.!?]+', context)
    question_words = set(re.findall(r'\w+', question_lower))
    
    best_sentence = ""
    best_score = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 10:
            continue
        sentence_words = set(re.findall(r'\w+', sentence.lower()))
        overlap = len(question_words & sentence_words)
        if overlap > best_score:
            best_score = overlap
            best_sentence = sentence
    
    return best_sentence if best_sentence else "No relevant information found."

async def generate_answer(question: str, context: str) -> str:
    """Generate answer with API fallback chain"""
    # Try OpenAI first
    answer = await call_openai_api(question, context)
    if answer:
        return answer
    
    # Try Google if OpenAI fails
    answer = await call_google_api(question, context)
    if answer:
        return answer
    
    # Fallback to rule-based extraction
    return extract_answer_from_context(question, context)


# API Endpoints
@app.get("/")
async def root():
    return {"message": "Vercel-Compatible Document Query System is running"}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "bearer_token_set": bool(Config.BEARER_TOKEN),
        "huggingface_key_set": bool(Config.HUGGINGFACE_API_KEY),
        "timestamp": "2025-07-31"
    }

@app.get("/test-apis")
async def test_apis():
    """Test API connectivity"""
    results = {
        "openai_available": bool(Config.OPENAI_API_KEY),
        "google_available": bool(Config.GOOGLE_API_KEY),
        "huggingface_available": bool(Config.HUGGINGFACE_API_KEY),
    }
    
    # Quick test of embedding generation
    try:
        test_embedding = await get_embedding("test text")
        results["embedding_working"] = len(test_embedding) > 0
        results["embedding_size"] = len(test_embedding)
    except Exception as e:
        results["embedding_error"] = str(e)
    
    return results

@app.post("/hackrx/run", response_model=RunResponse)
async def run_submissions(req: RunRequest, http_req: Request):
    # Authentication
    auth_header = http_req.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication required")
    
    provided_token = auth_header.split(" ")[1]
    if not Config.BEARER_TOKEN:
        raise HTTPException(status_code=500, detail="Server misconfiguration: BEARER_TOKEN not set in environment.")
    if provided_token != Config.BEARER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid Bearer token")
    
    try:
        # Fetch and process document
        doc_text = await fetch_document(req.documents)
        if not doc_text or len(doc_text.strip()) < 10:
            raise HTTPException(400, "Failed to extract content from document")
        
        # Smart chunking
        chunks = smart_chunk_text(doc_text)
        
        # Create embeddings for chunks (batch process for efficiency)
        chunk_embeddings = []
        for chunk in chunks:
            try:
                embedding = await get_embedding(chunk)
                chunk_embeddings.append((chunk, embedding))
            except Exception:
                # Skip chunks that fail embedding
                continue
        
        # Process each question
        answers = []
        for question in req.questions:
            try:
                # Get question embedding
                question_embedding = await get_embedding(question)
                
                # Find most relevant chunks using cosine similarity
                similarities = []
                for chunk, chunk_embedding in chunk_embeddings:
                    if chunk_embedding and question_embedding:
                        similarity = cosine_similarity(question_embedding, chunk_embedding)
                        similarities.append((chunk, similarity))
                
                # Sort by similarity and get top chunks
                similarities.sort(key=lambda x: x[1], reverse=True)
                relevant_chunks = [chunk for chunk, sim in similarities[:Config.MAX_RELEVANT_CHUNKS] 
                                 if sim > Config.SIMILARITY_THRESHOLD]
                
                if relevant_chunks:
                    # Combine relevant chunks as context
                    context = " ".join(relevant_chunks)
                    # Generate answer using the most relevant context
                    answer = await generate_answer(question, context)
                    answers.append(answer)
                else:
                    # Fallback: use simple text search
                    question_words = set(re.findall(r'\w+', question.lower()))
                    best_chunk = ""
                    best_score = 0
                    
                    for chunk in chunks:
                        chunk_words = set(re.findall(r'\w+', chunk.lower()))
                        overlap = len(question_words & chunk_words)
                        if overlap > best_score:
                            best_score = overlap
                            best_chunk = chunk
                    
                    if best_chunk:
                        answer = await generate_answer(question, best_chunk)
                        answers.append(answer)
                    else:
                        answers.append("I couldn't find relevant information to answer this question.")
                        
            except Exception as e:
                answers.append(f"Error processing question: {str(e)}")
        
        return RunResponse(answers=answers)
        
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Processing failed: {e}")
        raise HTTPException(500, f"An internal error occurred: {e}")

# For Vercel
app = app
