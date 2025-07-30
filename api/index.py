"""
RAG-based Document Query System for Vercel Deployment
"""

import os
import json
import re
import asyncio
import io
import hashlib
import sqlite3
import math
from datetime import datetime
from typing import List, Dict, Any

import httpx
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field

# Try to import numpy, but provide fallback
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# PDF Processing
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

app = FastAPI(title="RAG Document Query System", version="5.0.0")

# Configuration
class Config:
    BEARER_TOKEN = os.getenv("BEARER_TOKEN", "ca6914a6c8df9d1ce075149c3ab9f060e666c75940576e37a98b3cf0e9092c72")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
    
    OPENAI_MODEL = "gpt-3.5-turbo"
    OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"
    ANTHROPIC_MODEL = "claude-3-haiku-20240307"
    HUGGINGFACE_MODEL = "google/flan-t5-large"
    HUGGINGFACE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100
    MAX_CHUNKS = 10
    TOP_K_CHUNKS = 3
    
    REQUEST_TIMEOUT = 30.0
    MAX_DOCUMENT_SIZE = 10 * 1024 * 1024
    
    DB_FILE = "/tmp/document_embeddings.db"

# Vector Database
class VectorDB:
    def __init__(self, db_path: str = Config.DB_FILE):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS document_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_hash TEXT,
                chunk_id INTEGER,
                content TEXT,
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_document_hash ON document_chunks(document_hash)')
        conn.commit()
        conn.close()
    
    def store_chunks(self, document_hash: str, chunks: List[str], embeddings: List[List[float]]):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM document_chunks WHERE document_hash = ?', (document_hash,))
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            if NUMPY_AVAILABLE:
                embedding_blob = np.array(embedding, dtype=np.float32).tobytes()
            else:
                embedding_blob = json.dumps(embedding).encode('utf-8')
            cursor.execute('INSERT INTO document_chunks (document_hash, chunk_id, content, embedding) VALUES (?, ?, ?, ?)',
                           (document_hash, i, chunk, embedding_blob))
        conn.commit()
        conn.close()
    
    def get_similar_chunks(self, document_hash: str, query_embedding: List[float], top_k: int = Config.TOP_K_CHUNKS) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT chunk_id, content, embedding FROM document_chunks WHERE document_hash = ?', (document_hash,))
        all_chunks = cursor.fetchall()
        conn.close()
        
        if not all_chunks:
            return []
        
        similarities = []
        for chunk_id, content, embedding_blob in all_chunks:
            if NUMPY_AVAILABLE:
                try:
                    chunk_embedding = np.frombuffer(embedding_blob, dtype=np.float32).tolist()
                except:
                    chunk_embedding = json.loads(embedding_blob.decode('utf-8'))
            else:
                chunk_embedding = json.loads(embedding_blob.decode('utf-8'))
            
            similarity = self.cosine_similarity(query_embedding, chunk_embedding)
            similarities.append({'chunk_id': chunk_id, 'content': content, 'similarity': similarity})
        
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        if NUMPY_AVAILABLE:
            v1 = np.array(vec1)
            v2 = np.array(vec2)
            epsilon = 1e-9
            return float(np.dot(v1, v2) / ((np.linalg.norm(v1) * np.linalg.norm(v2)) + epsilon))
        else:
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = math.sqrt(sum(a * a for a in vec1))
            magnitude2 = math.sqrt(sum(b * b for b in vec2))
            epsilon = 1e-9
            return dot_product / ((magnitude1 * magnitude2) + epsilon) if magnitude1 != 0 and magnitude2 != 0 else 0.0
    
    def document_exists(self, document_hash: str) -> bool:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM document_chunks WHERE document_hash = ?', (document_hash,))
        count = cursor.fetchone()[0]
        conn.close()
        return count > 0

vector_db = VectorDB()

# Models
class RunRequest(BaseModel):
    documents: str = Field(..., description="URL to document")
    questions: List[str] = Field(..., description="Questions")

class RunResponse(BaseModel):
    answers: List[str] = Field(..., description="List of answers")

# Document Processing
async def fetch_document(url: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=Config.REQUEST_TIMEOUT) as client:
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

# Embeddings
async def get_embeddings(texts: List[str]) -> List[List[float]]:
    if Config.OPENAI_API_KEY:
        try:
            return await get_openai_embeddings(texts)
        except Exception as e:
            print(f"OpenAI embeddings failed: {e}")
    if Config.HUGGINGFACE_API_KEY:
        try:
            return await get_huggingface_embeddings(texts)
        except Exception as e:
            print(f"HuggingFace embeddings failed: {e}")
    return create_simple_embeddings(texts)

async def get_openai_embeddings(texts: List[str]) -> List[List[float]]:
    async with httpx.AsyncClient() as client:
        response = await client.post("https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {Config.OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={"model": Config.OPENAI_EMBEDDING_MODEL, "input": texts}, timeout=30.0)
        response.raise_for_status()
        return [item["embedding"] for item in response.json()["data"]]

async def get_huggingface_embeddings(texts: List[str]) -> List[List[float]]:
    async with httpx.AsyncClient() as client:
        response = await client.post(f"https://api-inference.huggingface.co/pipeline/feature-extraction/{Config.HUGGINGFACE_EMBEDDING_MODEL}",
            headers={"Authorization": f"Bearer {Config.HUGGINGFACE_API_KEY}", "Content-Type": "application/json"},
            json={"inputs": texts, "options": {"wait_for_model": True}}, timeout=30.0)
        response.raise_for_status()
        result = response.json()
        if isinstance(result, list) and result and isinstance(result[0], list):
            return result
        raise Exception("Unexpected embedding format from HuggingFace")

def create_simple_embeddings(texts: List[str]) -> List[List[float]]:
    keywords = ['prime', 'minister', 'health', 'family', 'welfare', 'waiting', 'period', 'days', 'months', 'maternity', 'pregnancy', 'coverage', 'benefit', 'cataract', 'surgery', 'treatment', 'organ', 'donor', 'hospital', 'ayush', 'room', 'rent', 'discount', 'claim', 'ncd', 'check', 'up', 'define', 'means', 'policy', 'sum', 'insured', 'premium', 'deductible']
    all_words = set(keywords)
    text_words = [re.findall(r'\w+', text.lower()) for text in texts]
    for words in text_words:
        all_words.update(words)
    vocab = sorted(list(all_words))[:512]
    
    embeddings = []
    for words in text_words:
        embedding = [0.0] * len(vocab)
        for word in words:
            if word in vocab:
                embedding[vocab.index(word)] += (3.0 if word in keywords else 1.0) / len(words)
        mag = math.sqrt(sum(x*x for x in embedding))
        if mag > 0:
            embedding = [x / mag for x in embedding]
        embeddings.append(embedding)
    return embeddings

# RAG Pipeline
async def process_document_for_rag(document_text: str) -> str:
    document_hash = hashlib.md5(document_text.encode()).hexdigest()
    if vector_db.document_exists(document_hash):
        return document_hash
    chunks = smart_chunk_text(document_text)
    embeddings = await get_embeddings(chunks)
    vector_db.store_chunks(document_hash, chunks, embeddings)
    return document_hash

async def call_llm_api(question: str, context_chunks: List[Dict[str, Any]], provider: str) -> Dict[str, Any]:
    context = "\n\n".join([f"[Relevance: {c['similarity']:.3f}] {c['content']}" for c in context_chunks])
    prompt = f"Based on the context below, answer the question.\n\nCONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
    
    if provider == "OpenAI" and Config.OPENAI_API_KEY:
        url, model, key = "https://api.openai.com/v1/chat/completions", Config.OPENAI_MODEL, Config.OPENAI_API_KEY
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 200, "temperature": 0.1}
    elif provider == "Anthropic" and Config.ANTHROPIC_API_KEY:
        url, model, key = "https://api.anthropic.com/v1/messages", Config.ANTHROPIC_MODEL, Config.ANTHROPIC_API_KEY
        payload = {"model": model, "max_tokens": 200, "messages": [{"role": "user", "content": prompt}]}
    elif provider == "HuggingFace" and Config.HUGGINGFACE_API_KEY:
        url, model, key = f"https://api-inference.huggingface.co/models/{Config.HUGGINGFACE_MODEL}", None, Config.HUGGINGFACE_API_KEY
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": 150, "return_full_text": False}}
    else:
        return {"success": False, "error": f"{provider} API key not available"}

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if provider == "Anthropic":
        headers.update({"x-api-key": key, "anthropic-version": "2023-06-01"})

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload, timeout=25.0)
            response.raise_for_status()
            result = response.json()
            
            if provider == "OpenAI":
                answer = result["choices"][0]["message"]["content"]
            elif provider == "Anthropic":
                answer = result["content"][0]["text"]
            else: # HuggingFace
                answer = result[0]["generated_text"] if isinstance(result, list) else str(result)
            
            return {"answer": answer.strip(), "success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}

def extract_answer_from_chunks(question: str, context_chunks: List[Dict[str, Any]]) -> str:
    if not context_chunks:
        return "No relevant information found."
    
    full_context = " ".join([c['content'] for c in context_chunks])
    sentences = re.split(r'(?<=[.!?])\s+', full_context)
    question_lower = question.lower()
    
    best_sentence, max_score = "", 0
    question_words = set(re.findall(r'\w{4,}', question_lower))

    for sentence in sentences:
        sentence_lower = sentence.lower()
        if any(skip in sentence_lower for skip in ['national insurance', 'uin:', 'cin -', 'page']):
            continue
        
        score = len(question_words.intersection(set(re.findall(r'\w+', sentence_lower))))
        if score > max_score:
            max_score = score
            best_sentence = sentence.strip()
            
    return best_sentence + "." if max_score > 1 else "Could not find a specific answer in the document."

async def generate_rag_answer(question: str, document_hash: str) -> str:
    try:
        question_embedding = (await get_embeddings([question]))[0]
        similar_chunks = vector_db.get_similar_chunks(document_hash, question_embedding)
        
        if not similar_chunks:
            return "No relevant information found."
        
        for provider in ["OpenAI", "Anthropic", "HuggingFace"]:
            result = await call_llm_api(question, similar_chunks, provider)
            if result.get("success"):
                return result["answer"]
            print(f"{provider} failed: {result.get('error')}")
        
        return extract_answer_from_chunks(question, similar_chunks)
    except Exception as e:
        print(f"RAG answer generation failed: {e}")
        return "Error processing the question."

# API Endpoints
@app.get("/")
async def root():
    return {"message": "Document Query System is running"}

@app.post("/hackrx/run", response_model=RunResponse)
async def run_submissions(req: RunRequest, http_req: Request):
    if http_req.headers.get("Authorization") != f"Bearer {Config.BEARER_TOKEN}":
        raise HTTPException(403, "Invalid Bearer token")
    
    try:
        doc_text = await fetch_document(req.documents)
        if not doc_text or len(doc_text.strip()) < 10:
            raise HTTPException(400, "Failed to extract content from document")
        
        doc_hash = await process_document_for_rag(doc_text)
        answers = [await generate_rag_answer(q, doc_hash) for q in req.questions]
        
        return RunResponse(answers=answers)
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Processing failed: {e}")
        raise HTTPException(500, f"An internal error occurred: {e}")

# For Vercel
app = app
