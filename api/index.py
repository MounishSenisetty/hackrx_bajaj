"""
Multi-LLM API with fallback providers: OpenAI, Anthropic Claude, and Hugging Face
"""

import os
import json
import re
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
import httpx
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Multi-LLM Query System", version="2.0.0")

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
    HUGGINGFACE_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    
    REQUEST_TIMEOUT = 30.0
    MAX_DOCUMENT_SIZE = 10 * 1024 * 1024

# Request/Response Models
class RunRequest(BaseModel):
    documents: str = Field(..., description="URL to document")
    questions: List[str] = Field(..., description="Questions")

class AnswerResponse(BaseModel):
    answer: str = Field(..., description="Answer")
    confidence_score: float = Field(..., description="Confidence score")
    relevant_chunks: List[Dict[str, Any]] = Field(..., description="Relevant chunks")
    reasoning: str = Field(..., description="Reasoning")
    llm_provider: str = Field(..., description="LLM provider used")

class RunResponse(BaseModel):
    answers: List[str] = Field(..., description="Answers")
    detailed_responses: Optional[List[AnswerResponse]] = Field(None, description="Detailed responses")
    processing_stats: Dict[str, Any] = Field(..., description="Processing stats")

# Document Processing
async def fetch_document(url: str) -> str:
    """Fetch and extract text from document URL."""
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
                return "Document content could not be decoded"
                
    except Exception as e:
        raise HTTPException(500, f"Document fetch failed: {str(e)}")

def chunk_text(text: str) -> List[Dict[str, Any]]:
    """Simple text chunking."""
    chunk_size = 800
    chunks = []
    words = text.split()
    
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i + chunk_size]
        chunk_text = ' '.join(chunk_words)
        
        chunks.append({
            'text': chunk_text,
            'score': 0.5,
            'metadata': {'chunk_id': len(chunks)}
        })
    
    return chunks

def simple_search(query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Simple keyword-based search."""
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
    
    scored_chunks.sort(key=lambda x: x['score'], reverse=True)
    return scored_chunks[:3]

# LLM Providers
async def call_openai(question: str, context: str) -> Dict[str, Any]:
    """Call OpenAI GPT API."""
    if not Config.OPENAI_API_KEY:
        raise Exception("OpenAI API key not available")
    
    prompt = f"""Based on the following context, answer the question concisely:

Context: {context[:2000]}

Question: {question}

Answer:"""

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
                    "max_tokens": 500,
                    "temperature": 0.3
                },
                timeout=20.0
            )
            response.raise_for_status()
            
            result = response.json()
            answer = result["choices"][0]["message"]["content"].strip()
            
            return {
                "answer": answer,
                "provider": "openai",
                "model": Config.OPENAI_MODEL,
                "success": True
            }
            
    except Exception as e:
        return {"success": False, "error": str(e), "provider": "openai"}

async def call_anthropic(question: str, context: str) -> Dict[str, Any]:
    """Call Anthropic Claude API."""
    if not Config.ANTHROPIC_API_KEY:
        raise Exception("Anthropic API key not available")
    
    prompt = f"""Based on the following context, answer the question concisely:

Context: {context[:2000]}

Question: {question}"""

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
                    "max_tokens": 500,
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=20.0
            )
            response.raise_for_status()
            
            result = response.json()
            answer = result["content"][0]["text"].strip()
            
            return {
                "answer": answer,
                "provider": "anthropic",
                "model": Config.ANTHROPIC_MODEL,
                "success": True
            }
            
    except Exception as e:
        return {"success": False, "error": str(e), "provider": "anthropic"}

async def call_huggingface(question: str, context: str) -> Dict[str, Any]:
    """Call Hugging Face Inference API."""
    if not Config.HUGGINGFACE_API_KEY or Config.HUGGINGFACE_API_KEY == "hf_demo_key_public":
        # Use a free model that doesn't require API key
        model_url = "https://api-inference.huggingface.co/models/google/flan-t5-large"
        headers = {"Content-Type": "application/json"}
    else:
        model_url = f"https://api-inference.huggingface.co/models/{Config.HUGGINGFACE_MODEL}"
        headers = {
            "Authorization": f"Bearer {Config.HUGGINGFACE_API_KEY}",
            "Content-Type": "application/json"
        }
    
    # Create a more structured prompt for better results
    prompt = f"""Based on the following context, answer the question concisely and accurately.

Context: {context[:1500]}

Question: {question}

Answer:"""

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                model_url,
                headers=headers,
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 150,
                        "temperature": 0.2,
                        "do_sample": True,
                        "top_p": 0.9
                    }
                },
                timeout=25.0
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Parse response based on model type
            if isinstance(result, list) and len(result) > 0:
                if "generated_text" in result[0]:
                    # For text generation models
                    full_text = result[0]["generated_text"]
                    # Extract only the new generated part after the prompt
                    if prompt in full_text:
                        answer = full_text.replace(prompt, "").strip()
                    else:
                        answer = full_text.strip()
                else:
                    # For other model types
                    answer = str(result[0])
            elif isinstance(result, dict):
                if "generated_text" in result:
                    answer = result["generated_text"].replace(prompt, "").strip()
                else:
                    answer = str(result)
            else:
                answer = str(result)
            
            # Clean up the answer
            if not answer or len(answer.strip()) < 3:
                answer = "Unable to generate a meaningful response from Hugging Face model."
            
            # Determine which model was actually used
            model_used = "google/flan-t5-large" if Config.HUGGINGFACE_API_KEY == "hf_demo_key_public" else Config.HUGGINGFACE_MODEL
            
            return {
                "answer": answer[:500],  # Limit answer length
                "provider": "huggingface",
                "model": model_used,
                "success": True
            }
            
    except Exception as e:
        return {"success": False, "error": str(e), "provider": "huggingface"}

def generate_fallback_answer(question: str, relevant_chunks: List[Dict[str, Any]]) -> str:
    """Generate a fallback answer using simple text processing."""
    if not relevant_chunks:
        return "I couldn't find relevant information in the document to answer this question."
    
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
        return answer
    else:
        # Fallback to first part of the chunk
        answer = context[:400].strip()
        if len(context) > 400:
            answer += "..."
        return answer

async def generate_answer_with_fallback(question: str, relevant_chunks: List[Dict[str, Any]]) -> AnswerResponse:
    """Generate answer using multiple LLM providers with fallback."""
    if not relevant_chunks:
        return AnswerResponse(
            answer="No relevant information found in the document.",
            confidence_score=0.0,
            relevant_chunks=[],
            reasoning="No relevant content found for the question.",
            llm_provider="none"
        )
    
    context = " ".join([chunk['text'] for chunk in relevant_chunks])
    
    # Try LLM providers in order: OpenAI -> Anthropic -> Hugging Face -> Local fallback
    llm_providers = [
        ("openai", call_openai),
        ("anthropic", call_anthropic),
        ("huggingface", call_huggingface)
    ]
    
    for provider_name, provider_func in llm_providers:
        try:
            result = await provider_func(question, context)
            if result.get("success"):
                return AnswerResponse(
                    answer=result["answer"],
                    confidence_score=0.8,
                    relevant_chunks=relevant_chunks,
                    reasoning=f"Answer generated using {result['provider']} model {result.get('model', 'unknown')}",
                    llm_provider=result["provider"]
                )
        except Exception as e:
            print(f"Provider {provider_name} failed: {e}")
            continue
    
    # Fallback to local processing
    fallback_answer = generate_fallback_answer(question, relevant_chunks)
    return AnswerResponse(
        answer=fallback_answer,
        confidence_score=0.5,
        relevant_chunks=relevant_chunks,
        reasoning="Answer generated using local text processing as all LLM providers were unavailable",
        llm_provider="local_fallback"
    )

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Multi-LLM Query System",
        "status": "operational",
        "available_providers": ["openai", "anthropic", "huggingface", "local_fallback"],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health():
    # Check which LLM providers are available
    providers_status = {
        "openai": bool(Config.OPENAI_API_KEY),
        "anthropic": bool(Config.ANTHROPIC_API_KEY),
        "huggingface": bool(Config.HUGGINGFACE_API_KEY),
        "local_fallback": True
    }
    
    return {
        "status": "healthy",
        "providers_available": providers_status,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/hackrx/run", response_model=RunResponse)
async def run_submissions(request_body: RunRequest, request: Request):
    """Main API endpoint with multi-LLM support."""
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
        document_text = await fetch_document(request_body.documents)
        
        if not document_text or len(document_text.strip()) < 10:
            raise HTTPException(400, "Could not extract meaningful content from document")
        
        # Chunk text
        chunks = chunk_text(document_text)
        
        # Process questions
        answers = []
        detailed_responses = []
        
        for question in request_body.questions:
            # Search for relevant chunks
            relevant_chunks = simple_search(question, chunks)
            
            # Generate answer with LLM fallback
            detailed_response = await generate_answer_with_fallback(question, relevant_chunks)
            
            answers.append(detailed_response.answer)
            detailed_responses.append(detailed_response)
        
        # Statistics
        processing_time = time.time() - start_time
        stats = {
            "processing_time_seconds": round(processing_time, 2),
            "document_length": len(document_text),
            "total_chunks": len(chunks),
            "questions_processed": len(request_body.questions),
            "environment": "vercel_multi_llm",
            "providers_attempted": [resp.llm_provider for resp in detailed_responses],
            "timestamp": datetime.now().isoformat()
        }
        
        return RunResponse(
            answers=answers,
            detailed_responses=detailed_responses,
            processing_stats=stats
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Processing failed: {str(e)}")

# For Vercel
app = app
