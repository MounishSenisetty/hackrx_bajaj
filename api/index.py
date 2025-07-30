"""
Simplified Document Query System for Vercel Deployment
"""

import os
import json
import re
import asyncio
import io
from datetime import datetime
from typing import List, Dict, Any, Optional
import httpx
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field

# PDF Processing
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

app = FastAPI(title="Document Query System", version="4.0.0")

# Configuration
class Config:
    BEARER_TOKEN = "ca6914a6c8df9d1ce075149c3ab9f060e666c75940576e37a98b3cf0e9092c72"
    
    # API Keys - prioritize environment variables
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
    
    # LLM Models and endpoints
    OPENAI_MODEL = "gpt-3.5-turbo"
    ANTHROPIC_MODEL = "claude-3-haiku-20240307"
    HUGGINGFACE_MODEL = "google/flan-t5-large"
    
    # Processing Configuration
    CHUNK_SIZE = 1000  # Characters for simple chunking
    CHUNK_OVERLAP = 200
    MAX_CHUNKS = 5
    
    REQUEST_TIMEOUT = 30.0
    MAX_DOCUMENT_SIZE = 10 * 1024 * 1024  # 10MB limit for Vercel

# Request/Response Models
class RunRequest(BaseModel):
    documents: str = Field(..., description="URL to document")
    questions: List[str] = Field(..., description="Questions")

class RunResponse(BaseModel):
    answers: List[str] = Field(..., description="List of answers corresponding to the input questions")

# Document Processing
async def fetch_document(url: str) -> str:
    """Fetch and extract text from document URL with proper PDF support."""
    try:
        async with httpx.AsyncClient(timeout=Config.REQUEST_TIMEOUT) as client:
            response = await client.get(url)
            response.raise_for_status()
            
            content = response.content
            if len(content) > Config.MAX_DOCUMENT_SIZE:
                raise HTTPException(400, "Document exceeds size limit")
            
            # Check if it's a PDF by content type or URL
            content_type = response.headers.get('content-type', '').lower()
            is_pdf = 'pdf' in content_type or url.lower().endswith('.pdf')
            
            if is_pdf and PDF_AVAILABLE:
                return extract_pdf_text(content)
            else:
                return extract_text_fallback(content)
                
    except Exception as e:
        raise HTTPException(500, f"Document fetch failed: {str(e)}")

def extract_pdf_text(content: bytes) -> str:
    """Extract text from PDF content using PyPDF2."""
    try:
        pdf_file = io.BytesIO(content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        
        # Clean up the extracted text
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        if len(text) < 50:
            # If PDF extraction failed, try fallback
            return extract_text_fallback(content)
        
        return text
        
    except Exception as e:
        print(f"PDF extraction failed: {e}")
        return extract_text_fallback(content)

def extract_text_fallback(content: bytes) -> str:
    """Fallback text extraction for non-PDF or when PDF extraction fails."""
    try:
        # Try UTF-8 first
        text = content.decode('utf-8', errors='ignore')
        
        # Clean up common PDF artifacts if this is raw PDF content
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        if len(text) < 100:
            # Try latin-1 encoding
            text = content.decode('latin-1', errors='ignore')
            text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
        
        return text if text else "Document content could not be decoded"
        
    except Exception:
        return "Document content could not be decoded"

def simple_chunk_text(text: str) -> List[str]:
    """Simple text chunking for processing."""
    if len(text) <= Config.CHUNK_SIZE:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + Config.CHUNK_SIZE
        
        # Try to end at sentence boundary
        if end < len(text):
            # Look for sentence endings within overlap distance
            for i in range(end, max(start + Config.CHUNK_SIZE - Config.CHUNK_OVERLAP, end - 200), -1):
                if text[i] in '.!?':
                    end = i + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - Config.CHUNK_OVERLAP
        
        if len(chunks) >= Config.MAX_CHUNKS:
            break
    
    return chunks

def simple_search(question: str, chunks: List[str]) -> List[str]:
    """Simple keyword-based search to find relevant chunks."""
    question_words = set(re.findall(r'\w+', question.lower()))
    scored_chunks = []
    
    for chunk in chunks:
        chunk_words = set(re.findall(r'\w+', chunk.lower()))
        overlap = len(question_words.intersection(chunk_words))
        
        if overlap > 0:
            score = overlap / len(question_words)
            
            # Boost for exact phrases
            if len(question) > 10 and question.lower() in chunk.lower():
                score += 0.5
            
            # Boost for numbers if question contains numbers
            if re.search(r'\d+', question) and re.search(r'\d+', chunk):
                score += 0.2
            
            scored_chunks.append((score, chunk))
    
    # Sort by score and return top chunks
    scored_chunks.sort(reverse=True)
    return [chunk for score, chunk in scored_chunks[:3]]

async def call_openai(question: str, context: str) -> Dict[str, Any]:
    """Call OpenAI GPT API for intelligent answer extraction."""
    if not Config.OPENAI_API_KEY:
        raise Exception("OpenAI API key not available")

    prompt = f"""Extract the precise answer to the question from the document context.

CONTEXT:
{context[:3000]}

QUESTION: {question}

INSTRUCTIONS:
- Find the specific information that answers the question
- Use exact numbers, dates, and terms from the document
- Provide a clear, factual response
- If not found, state that clearly

ANSWER:"""

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {Config.OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": Config.OPENAI_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 150,
                    "temperature": 0.1
                },
                timeout=20.0
            )
            response.raise_for_status()
            
            result = response.json()
            answer = result["choices"][0]["message"]["content"].strip()
            
            return {"answer": answer, "success": True}
            
    except Exception as e:
        return {"success": False, "error": str(e)}

async def call_anthropic(question: str, context: str) -> Dict[str, Any]:
    """Call Anthropic Claude API for intelligent answer extraction."""
    if not Config.ANTHROPIC_API_KEY:
        raise Exception("Anthropic API key not available")
    
    prompt = f"""Extract the precise answer to the question from the document context.

CONTEXT:
{context[:3000]}

QUESTION: {question}

INSTRUCTIONS:
- Find the specific information that answers the question
- Use exact numbers, dates, and terms from the document
- Provide a clear, factual response
- If not found, state that clearly

ANSWER:"""

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": Config.ANTHROPIC_API_KEY,
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01"
                },
                json={
                    "model": Config.ANTHROPIC_MODEL,
                    "max_tokens": 150,
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=20.0
            )
            response.raise_for_status()
            
            result = response.json()
            answer = result["content"][0]["text"].strip()
            
            return {"answer": answer, "success": True}
            
    except Exception as e:
        return {"success": False, "error": str(e)}

async def call_huggingface(question: str, context: str) -> Dict[str, Any]:
    """Call Hugging Face API for answer extraction."""
    try:
        prompt = f"Question: {question}\n\nContext: {context[:2000]}\n\nAnswer:"
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://api-inference.huggingface.co/models/{Config.HUGGINGFACE_MODEL}",
                headers={"Content-Type": "application/json"},
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 100,
                        "temperature": 0.1,
                        "return_full_text": False
                    }
                },
                timeout=25.0
            )
            response.raise_for_status()
            
            result = response.json()
            
            if isinstance(result, list) and len(result) > 0:
                answer = result[0].get("generated_text", "").strip()
            else:
                answer = str(result).strip()
            
            if answer and len(answer) > 10:
                return {"answer": answer[:200], "success": True}
            else:
                return {"success": False, "error": "No meaningful response"}
            
    except Exception as e:
        return {"success": False, "error": str(e)}

def extract_answer_from_text(question: str, chunks: List[str]) -> str:
    """Extract answer using simple text processing."""
    if not chunks:
        return "No relevant information found in the document."
    
    # Combine relevant chunks
    combined_text = " ".join(chunks)
    
    # Find sentences most relevant to the question
    sentences = re.split(r'[.!?]+', combined_text)
    question_words = set(re.findall(r'\w+', question.lower()))
    
    # Remove common stop words
    stop_words = {'what', 'is', 'the', 'are', 'does', 'do', 'can', 'will', 'how', 'when', 'where', 'why', 'which'}
    key_words = question_words - stop_words
    
    best_sentence = ""
    best_score = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 15:
            continue
        
        sentence_words = set(re.findall(r'\w+', sentence.lower()))
        overlap = len(key_words.intersection(sentence_words))
        
        score = overlap / max(len(key_words), 1) if key_words else 0
        
        # Boost for numbers
        if re.search(r'\d+', sentence):
            score += 0.2
        
        if score > best_score:
            best_score = score
            best_sentence = sentence
    
    if best_sentence and best_score > 0.1:
        if not best_sentence.endswith(('.', '!', '?')):
            best_sentence += '.'
        return best_sentence
    
    # Fallback to first substantial sentence
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 30:
            if not sentence.endswith(('.', '!', '?')):
                sentence += '.'
            return sentence
    
    return "The requested information is available in the document but requires more specific context."

async def generate_answer(question: str, document_text: str) -> str:
    """Generate answer using available LLM providers with fallbacks."""
    try:
        # Chunk the document
        chunks = simple_chunk_text(document_text)
        
        # Find relevant chunks
        relevant_chunks = simple_search(question, chunks)
        
        if not relevant_chunks:
            return "No relevant information found in the document."
        
        context = " ".join(relevant_chunks)
        
        # Try LLM providers in order
        providers = [
            ("OpenAI", call_openai),
            ("Anthropic", call_anthropic),
            ("HuggingFace", call_huggingface)
        ]
        
        for provider_name, provider_func in providers:
            try:
                result = await provider_func(question, context)
                if result.get("success"):
                    return result["answer"]
            except Exception as e:
                print(f"{provider_name} failed: {e}")
                continue
        
        # Final fallback to local processing
        return extract_answer_from_text(question, relevant_chunks)
        
    except Exception as e:
        print(f"Answer generation failed: {e}")
        return "Unable to process the question at this time."

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Document Query System",
        "status": "operational",
        "endpoint": "/hackrx/run",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "pdf_support": PDF_AVAILABLE,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/hackrx/run", response_model=RunResponse)
async def run_submissions(request_body: RunRequest, request: Request):
    """Main API endpoint for document-based question answering."""
    
    # Authentication
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication required")
    
    provided_token = auth_header.split(" ")[1]
    if provided_token != Config.BEARER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid Bearer token")
    
    try:
        # Process document
        document_text = await fetch_document(request_body.documents)
        
        if not document_text or len(document_text.strip()) < 10:
            raise HTTPException(400, "Could not extract meaningful content from document")
        
        # Process questions and generate answers
        answers = []
        
        for question in request_body.questions:
            answer = await generate_answer(question, document_text)
            answers.append(answer)
        
        return RunResponse(answers=answers)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Processing failed: {str(e)}")

# For Vercel
app = app
