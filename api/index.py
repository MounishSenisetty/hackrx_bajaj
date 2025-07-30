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
# LangChain imports
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

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

# Sentence Transformer for local embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False

app = FastAPI(title="RAG Document Query System", version="5.0.0")

# Configuration
class Config:
    BEARER_TOKEN = os.getenv("BEARER_TOKEN", "ca6914a6c8df9d1ce075149c3ab9f060e666c75940576e37a98b3cf0e9092c72")
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
    
    HUGGINGFACE_MODEL = "google/flan-t5-base"
    HUGGINGFACE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100
    MAX_CHUNKS = 10
    TOP_K_CHUNKS = 5
    
    REQUEST_TIMEOUT = 30.0
    MAX_DOCUMENT_SIZE = 10 * 1024 * 1024
    
    DB_FILE = "/tmp/document_embeddings.db"


# LangChain Embeddings and LLM setup
embedding_model = HuggingFaceEmbeddings(
    model_name=Config.HUGGINGFACE_EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"}
)

llm = HuggingFaceHub(
    repo_id=Config.HUGGINGFACE_MODEL,
    huggingfacehub_api_token=Config.HUGGINGFACE_API_KEY,
    model_kwargs={"max_new_tokens": 250}
)

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


# LangChain Embedding function
def get_langchain_embeddings(texts: List[str]) -> List[List[float]]:
    return embedding_model.embed_documents(texts)

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
                embedding[vocab.index(word)] += (3.0 if word in keywords else 1.0) / (len(words) + 1e-6)
        mag = math.sqrt(sum(x*x for x in embedding))
        if mag > 0:
            embedding = [x / mag for x in embedding]
        embeddings.append(embedding)
    return embeddings


# LangChain RAG Pipeline
def process_document_for_rag(document_text: str):
    # Use LangChain's text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )
    docs = text_splitter.create_documents([document_text])
    # Create FAISS vector store
    vectorstore = FAISS.from_documents(docs, embedding_model)
    return vectorstore


def get_qa_chain(vectorstore):
    # Custom prompt (optional, can use default)
    prompt_template = (
        "You are an expert Q&A assistant. Your goal is to provide accurate and helpful answers. "
        "Please follow these instructions carefully:\n"
        "1. Analyze the user's QUESTION and the provided CONTEXT snippets.\n"
        "2. Synthesize an answer based *primarily* on the information in the CONTEXT.\n"
        "3. If the CONTEXT does not contain enough information to answer the question, "
        "use your general knowledge to provide a helpful response.\n"
        "4. If you use your general knowledge, you MUST begin your answer with the phrase: "
        "'Based on my general knowledge, as the document does not contain specific details on this topic...'\n\n"
        "CONTEXT:\n---\n{context}\n---\n\n"
        "QUESTION: {question}\n\n"
        "ANSWER:"
    )
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": PROMPT}
    )


def generate_rag_answer(question: str, vectorstore) -> str:
    try:
        qa_chain = get_qa_chain(vectorstore)
        result = qa_chain({"query": question})
        return result["result"]
    except Exception as e:
        print(f"LangChain RAG answer generation failed: {e}")
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
        vectorstore = process_document_for_rag(doc_text)
        answers = [generate_rag_answer(q, vectorstore) for q in req.questions]
        return RunResponse(answers=answers)
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Processing failed: {e}")
        raise HTTPException(500, f"An internal error occurred: {e}")

# For Vercel
app = app
