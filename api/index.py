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
from typing import List, Dict, Any, Optional
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
    BEARER_TOKEN = "ca6914a6c8df9d1ce075149c3ab9f060e666c75940576e37a98b3cf0e9092c72"
    
    # API Keys - prioritize environment variables
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
    
    # LLM Models and endpoints
    OPENAI_MODEL = "gpt-3.5-turbo"
    OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"
    ANTHROPIC_MODEL = "claude-3-haiku-20240307"
    HUGGINGFACE_MODEL = "google/flan-t5-large"
    HUGGINGFACE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    # RAG Configuration
    CHUNK_SIZE = 500  # Smaller chunks for better semantic matching
    CHUNK_OVERLAP = 100
    MAX_CHUNKS = 10
    TOP_K_CHUNKS = 3  # Number of most relevant chunks to use for answer
    
    REQUEST_TIMEOUT = 30.0
    MAX_DOCUMENT_SIZE = 10 * 1024 * 1024  # 10MB limit for Vercel
    
    # Database file for storing embeddings
    DB_FILE = "/tmp/document_embeddings.db"

# Vector Database Management
class VectorDB:
    def __init__(self, db_path: str = Config.DB_FILE):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize SQLite database for storing embeddings."""
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
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_document_hash 
            ON document_chunks(document_hash)
        ''')
        conn.commit()
        conn.close()
    
    def store_chunks(self, document_hash: str, chunks: List[str], embeddings: List[List[float]]):
        """Store document chunks and their embeddings."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Clear existing chunks for this document
        cursor.execute('DELETE FROM document_chunks WHERE document_hash = ?', (document_hash,))
        
        # Insert new chunks
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Convert embedding to binary format
            if NUMPY_AVAILABLE:
                embedding_blob = np.array(embedding, dtype=np.float32).tobytes()
            else:
                # Simple binary encoding for list of floats
                embedding_blob = json.dumps(embedding).encode('utf-8')
            
            cursor.execute('''
                INSERT INTO document_chunks (document_hash, chunk_id, content, embedding)
                VALUES (?, ?, ?, ?)
            ''', (document_hash, i, chunk, embedding_blob))
        
        conn.commit()
        conn.close()
    
    def get_similar_chunks(self, document_hash: str, query_embedding: List[float], top_k: int = Config.TOP_K_CHUNKS) -> List[Dict[str, Any]]:
        """Retrieve most similar chunks using cosine similarity."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT chunk_id, content, embedding 
            FROM document_chunks 
            WHERE document_hash = ?
        ''', (document_hash,))
        
        chunks = cursor.fetchall()
        conn.close()
        
        if not chunks:
            return []
        
        similarities = []
        
        for chunk_id, content, embedding_blob in chunks:
            # Skip chunks that are mostly header/footer information
            content_lower = content.lower()
            if any(skip_term in content_lower for skip_term in [
                'national insurance company limited',
                'uin:', 'cin -', 'irdai regn',
                'page', 'kolkata', 'new town'
            ]) and len(content) < 200:
                continue
            
            # Decode embedding
            if NUMPY_AVAILABLE:
                try:
                    chunk_embedding = np.frombuffer(embedding_blob, dtype=np.float32).tolist()
                except:
                    chunk_embedding = json.loads(embedding_blob.decode('utf-8'))
            else:
                chunk_embedding = json.loads(embedding_blob.decode('utf-8'))
            
            # Calculate cosine similarity
            similarity = self.cosine_similarity(query_embedding, chunk_embedding)
            
            # Boost similarity for content-rich chunks
            if len(content) > 300:
                similarity += 0.1
            
            similarities.append({
                'chunk_id': chunk_id,
                'content': content,
                'similarity': similarity
            })
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors without numpy."""
        if NUMPY_AVAILABLE:
            # Use numpy for better performance
            v1 = np.array(vec1)
            v2 = np.array(vec2)
            return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        else:
            # Pure Python implementation
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = math.sqrt(sum(a * a for a in vec1))
            magnitude2 = math.sqrt(sum(b * b for b in vec2))
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            return dot_product / (magnitude1 * magnitude2)
    
    def document_exists(self, document_hash: str) -> bool:
        """Check if document is already processed and stored."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM document_chunks WHERE document_hash = ?', (document_hash,))
        count = cursor.fetchone()[0]
        conn.close()
        return count > 0

# Initialize vector database
vector_db = VectorDB()

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

def smart_chunk_text(text: str) -> List[str]:
    """Create semantic chunks based on content structure."""
    if len(text) <= Config.CHUNK_SIZE:
        return [text]
    
    # Split by paragraphs first
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # If adding this paragraph would exceed chunk size
        if len(current_chunk) + len(paragraph) > Config.CHUNK_SIZE:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
        
        # If paragraph itself is too long, split by sentences
        if len(paragraph) > Config.CHUNK_SIZE:
            sentences = re.split(r'[.!?]+', paragraph)
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                if len(current_chunk) + len(sentence) > Config.CHUNK_SIZE:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = sentence
                    else:
                        # Single sentence too long, force split
                        if len(sentence) > Config.CHUNK_SIZE:
                            words = sentence.split()
                            temp_chunk = ""
                            for word in words:
                                if len(temp_chunk) + len(word) > Config.CHUNK_SIZE:
                                    if temp_chunk:
                                        chunks.append(temp_chunk.strip())
                                        temp_chunk = word
                                    else:
                                        chunks.append(word)
                                else:
                                    temp_chunk += " " + word if temp_chunk else word
                            if temp_chunk:
                                current_chunk = temp_chunk
                        else:
                            current_chunk = sentence
                else:
                    current_chunk += ". " + sentence if current_chunk else sentence
        else:
            current_chunk += "\n\n" + paragraph if current_chunk else paragraph
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Add overlap between chunks
    overlapped_chunks = []
    for i, chunk in enumerate(chunks):
        if i > 0 and len(chunks[i-1]) > Config.CHUNK_OVERLAP:
            # Add overlap from previous chunk
            overlap = chunks[i-1][-Config.CHUNK_OVERLAP:]
            chunk = overlap + " " + chunk
        overlapped_chunks.append(chunk)
    
    return overlapped_chunks[:Config.MAX_CHUNKS]

async def get_embeddings(texts: List[str], is_query: bool = False) -> List[List[float]]:
    """Get embeddings for texts using available embedding models."""
    
    # Try OpenAI embeddings first
    if Config.OPENAI_API_KEY:
        try:
            return await get_openai_embeddings(texts)
        except Exception as e:
            print(f"OpenAI embeddings failed: {e}")
    
    # Fallback to HuggingFace embeddings
    try:
        return await get_huggingface_embeddings(texts)
    except Exception as e:
        print(f"HuggingFace embeddings failed: {e}")
    
    # Simple fallback: create pseudo-embeddings based on word frequency
    return create_simple_embeddings(texts)

async def get_openai_embeddings(texts: List[str]) -> List[List[float]]:
    """Get embeddings using OpenAI's text-embedding-ada-002."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.openai.com/v1/embeddings",
            headers={
                "Authorization": f"Bearer {Config.OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": Config.OPENAI_EMBEDDING_MODEL,
                "input": texts
            },
            timeout=30.0
        )
        response.raise_for_status()
        
        result = response.json()
        return [item["embedding"] for item in result["data"]]

async def get_huggingface_embeddings(texts: List[str]) -> List[List[float]]:
    """Get embeddings using HuggingFace sentence transformers."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"https://api-inference.huggingface.co/pipeline/feature-extraction/{Config.HUGGINGFACE_EMBEDDING_MODEL}",
            headers={"Content-Type": "application/json"},
            json={"inputs": texts},
            timeout=30.0
        )
        response.raise_for_status()
        
        result = response.json()
        
        # Handle different response formats
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], list) and isinstance(result[0][0], list):
                # Format: [[[embedding1]], [[embedding2]]]
                return [item[0] for item in result]
            elif isinstance(result[0], list):
                # Format: [[embedding1], [embedding2]]
                return result
        
        raise Exception("Unexpected embedding format")

def create_simple_embeddings(texts: List[str]) -> List[List[float]]:
    """Create better embeddings based on TF-IDF like approach with semantic keywords."""
    # Define important keywords for health insurance domain
    important_keywords = [
        'prime', 'minister', 'health', 'family', 'welfare', 'waiting', 'period', 'days', 'months',
        'maternity', 'pregnancy', 'coverage', 'benefit', 'cataract', 'surgery', 'treatment',
        'organ', 'donor', 'hospital', 'ayush', 'room', 'rent', 'discount', 'claim', 'ncd',
        'check', 'up', 'define', 'means', 'policy', 'sum', 'insured', 'premium', 'deductible'
    ]
    
    # Collect all words including important keywords
    all_words = set(important_keywords)  # Start with important keywords
    text_words = []
    
    for text in texts:
        words = re.findall(r'\w+', text.lower())
        text_words.append(words)
        all_words.update(words)
    
    vocab = sorted(list(all_words))
    vocab_size = min(len(vocab), 512)  # Increase dimension for better representation
    vocab = vocab[:vocab_size]
    
    embeddings = []
    for words in text_words:
        embedding = [0.0] * vocab_size
        word_count = len(words)
        
        for word in words:
            if word in vocab:
                idx = vocab.index(word)
                # Give higher weight to important keywords
                weight = 3.0 if word in important_keywords else 1.0
                embedding[idx] += weight / word_count
        
        # Normalize the embedding
        magnitude = math.sqrt(sum(x * x for x in embedding))
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
        
        embeddings.append(embedding)
    
    return embeddings

def get_document_hash(text: str) -> str:
    """Generate hash for document content."""
    return hashlib.md5(text.encode()).hexdigest()

async def process_document_for_rag(document_text: str) -> str:
    """Process document for RAG: chunk, embed, and store."""
    document_hash = get_document_hash(document_text)
    
    # Check if already processed
    if vector_db.document_exists(document_hash):
        print(f"Document {document_hash} already processed")
        return document_hash
    
    # Chunk the document
    chunks = smart_chunk_text(document_text)
    print(f"Created {len(chunks)} chunks for document")
    
    # Get embeddings for all chunks
    embeddings = await get_embeddings(chunks)
    print(f"Generated embeddings for {len(embeddings)} chunks")
    
    # Store in vector database
    vector_db.store_chunks(document_hash, chunks, embeddings)
    print(f"Stored document {document_hash} in vector database")
    
    return document_hash

async def call_openai_rag(question: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Call OpenAI GPT API for RAG-based answer generation."""
    if not Config.OPENAI_API_KEY:
        raise Exception("OpenAI API key not available")

    # Prepare context from retrieved chunks
    context_text = "\n\n".join([
        f"[Relevance: {chunk['similarity']:.3f}] {chunk['content']}" 
        for chunk in context_chunks
    ])

    prompt = f"""You are an expert document analyst. Based on the provided document context, answer the question accurately and comprehensively.

RETRIEVED CONTEXT:
{context_text}

QUESTION: {question}

INSTRUCTIONS:
- Analyze the provided context carefully
- Extract the specific information that answers the question
- Use exact numbers, dates, names, and terms from the document
- If the information is not in the context, clearly state that
- Provide a clear, factual, and complete response
- Be specific and precise in your answer

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
                    "max_tokens": 200,
                    "temperature": 0.1
                },
                timeout=25.0
            )
            response.raise_for_status()
            
            result = response.json()
            answer = result["choices"][0]["message"]["content"].strip()
            
            return {"answer": answer, "success": True}
            
    except Exception as e:
        return {"success": False, "error": str(e)}

async def call_anthropic_rag(question: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Call Anthropic Claude API for RAG-based answer generation."""
    if not Config.ANTHROPIC_API_KEY:
        raise Exception("Anthropic API key not available")
    
    context_text = "\n\n".join([
        f"[Relevance: {chunk['similarity']:.3f}] {chunk['content']}" 
        for chunk in context_chunks
    ])

    prompt = f"""You are an expert document analyst. Based on the provided document context, answer the question accurately and comprehensively.

RETRIEVED CONTEXT:
{context_text}

QUESTION: {question}

INSTRUCTIONS:
- Analyze the provided context carefully
- Extract the specific information that answers the question
- Use exact numbers, dates, names, and terms from the document
- If the information is not in the context, clearly state that
- Provide a clear, factual, and complete response
- Be specific and precise in your answer

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
                    "max_tokens": 200,
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=25.0
            )
            response.raise_for_status()
            
            result = response.json()
            answer = result["content"][0]["text"].strip()
            
            return {"answer": answer, "success": True}
            
    except Exception as e:
        return {"success": False, "error": str(e)}

async def call_huggingface_rag(question: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Call Hugging Face API for RAG-based answer generation."""
    try:
        context_text = " ".join([chunk['content'] for chunk in context_chunks[:2]])  # Limit context
        prompt = f"Context: {context_text[:1500]}\n\nQuestion: {question}\n\nAnswer based on the context:"
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://api-inference.huggingface.co/models/{Config.HUGGINGFACE_MODEL}",
                headers={"Content-Type": "application/json"},
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 150,
                        "temperature": 0.1,
                        "return_full_text": False
                    }
                },
                timeout=30.0
            )
            response.raise_for_status()
            
            result = response.json()
            
            if isinstance(result, list) and len(result) > 0:
                answer = result[0].get("generated_text", "").strip()
            else:
                answer = str(result).strip()
            
            if answer and len(answer) > 10:
                return {"answer": answer[:250], "success": True}
            else:
                return {"success": False, "error": "No meaningful response"}
            
    except Exception as e:
        return {"success": False, "error": str(e)}

def extract_answer_from_chunks(question: str, context_chunks: List[Dict[str, Any]]) -> str:
    """Enhanced fallback answer extraction when LLM APIs fail."""
    if not context_chunks:
        return "No relevant information found in the document."
    
    question_lower = question.lower()
    
    # Combine all relevant chunks for broader search
    all_content = " ".join([chunk['content'] for chunk in context_chunks])
    
    # Split content into sentences
    sentences = [s.strip() for s in re.split(r'[.!?]+', all_content) if len(s.strip()) > 15]
    
    # Enhanced question-specific patterns
    patterns = {
        'prime minister': {
            'keywords': ['prime minister', 'minister', 'health', 'family welfare'],
            'boost': 10
        },
        'waiting period': {
            'keywords': ['waiting period', 'waiting', 'period', 'days', 'months'],
            'boost': 8,
            'number_boost': True
        },
        'maternity': {
            'keywords': ['maternity', 'pregnancy', 'childbirth', 'delivery'],
            'boost': 8,
            'number_boost': True
        },
        'cataract': {
            'keywords': ['cataract', 'eye', 'surgery', 'lens'],
            'boost': 8
        },
        'organ donor': {
            'keywords': ['organ donor', 'organ', 'donor', 'transplant'],
            'boost': 7
        },
        'hospital': {
            'keywords': ['hospital', 'medical facility', 'healthcare'],
            'boost': 6,
            'definition': True
        },
        'ayush': {
            'keywords': ['ayush', 'alternative medicine', 'homeopathy', 'ayurveda'],
            'boost': 8
        },
        'room rent': {
            'keywords': ['room rent', 'room', 'accommodation', 'charges'],
            'boost': 7,
            'number_boost': True
        },
        'discount': {
            'keywords': ['discount', 'ncd', 'claim', 'reduction'],
            'boost': 6,
            'number_boost': True
        },
        'health check': {
            'keywords': ['health check', 'check up', 'preventive', 'screening'],
            'boost': 6
        }
    }
    
    best_sentences = []
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        total_score = 0
        
        # Check each pattern
        for pattern_key, pattern_info in patterns.items():
            if pattern_key in question_lower:
                pattern_score = 0
                
                # Check for keyword matches
                for keyword in pattern_info['keywords']:
                    if keyword in sentence_lower:
                        pattern_score += pattern_info['boost']
                        break
                
                # Boost for numbers if pattern expects them
                if pattern_info.get('number_boost') and re.search(r'\d+', sentence):
                    pattern_score += 3
                
                # Boost for definitions
                if pattern_info.get('definition') and ('means' in sentence_lower or 'defined' in sentence_lower or 'refers to' in sentence_lower):
                    pattern_score += 4
                
                total_score += pattern_score
        
        # General question word overlap
        question_words = set(re.findall(r'\w{4,}', question_lower))  # Only longer words
        sentence_words = set(re.findall(r'\w+', sentence_lower))
        overlap = len(question_words.intersection(sentence_words))
        total_score += overlap * 2
        
        # Penalize generic company information
        if 'national insurance' in sentence_lower and 'company limited' in sentence_lower:
            total_score -= 5
        
        if 'uin:' in sentence_lower or 'cin -' in sentence_lower or 'irdai regn' in sentence_lower:
            total_score -= 8
        
        if total_score > 3:  # Only consider sentences with reasonable scores
            best_sentences.append({
                'text': sentence,
                'score': total_score
            })
    
    # Sort by score and get the best answer
    best_sentences.sort(key=lambda x: x['score'], reverse=True)
    
    if best_sentences and best_sentences[0]['score'] > 5:
        answer = best_sentences[0]['text'].strip()
        if not answer.endswith(('.', '!', '?')):
            answer += "."
        return answer
    
    # If no high-scoring sentence, try to find any relevant content
    for sentence in sentences:
        sentence_lower = sentence.lower()
        
        # Skip header/footer information
        if any(skip in sentence_lower for skip in ['national insurance', 'kolkata', 'uin:', 'cin -', 'page']):
            continue
        
        # Look for sentences with question-relevant content
        question_words = question_lower.split()
        relevant_words = [w for w in question_words if len(w) > 3]
        
        for word in relevant_words:
            if word in sentence_lower and len(sentence) > 30:
                if not sentence.endswith(('.', '!', '?')):
                    sentence += "."
                return sentence
    
    return "The specific information requested is not clearly available in the document sections retrieved."

async def generate_rag_answer(question: str, document_hash: str) -> str:
    """Generate answer using RAG approach with vector similarity search."""
    try:
        # Get embeddings for the question
        print(f"Generating embeddings for question: {question}")
        question_embedding = await get_embeddings([question])
        if not question_embedding:
            return "Unable to process question for similarity search."
        
        # Retrieve similar chunks from vector database
        similar_chunks = vector_db.get_similar_chunks(
            document_hash, 
            question_embedding[0], 
            top_k=Config.TOP_K_CHUNKS
        )
        
        if not similar_chunks:
            return "No relevant information found for the question."
        
        print(f"Retrieved {len(similar_chunks)} similar chunks")
        for i, chunk in enumerate(similar_chunks):
            print(f"Chunk {i+1} similarity: {chunk['similarity']:.3f}, content preview: {chunk['content'][:100]}...")
        
        # Try LLM providers in order with RAG context
        providers = [
            ("OpenAI", call_openai_rag),
            ("Anthropic", call_anthropic_rag),
            ("HuggingFace", call_huggingface_rag)
        ]
        
        for provider_name, provider_func in providers:
            try:
                print(f"Trying {provider_name} for answer generation...")
                result = await provider_func(question, similar_chunks)
                if result.get("success"):
                    print(f"✓ Answer generated using {provider_name}")
                    return result["answer"]
                else:
                    print(f"✗ {provider_name} failed: {result.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"✗ {provider_name} exception: {e}")
                continue
        
        # Final fallback to rule-based extraction
        print("Using enhanced fallback extraction")
        answer = extract_answer_from_chunks(question, similar_chunks)
        print(f"Fallback answer: {answer}")
        return answer
        
    except Exception as e:
        print(f"RAG answer generation failed: {e}")
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
    """Main API endpoint for RAG-based document question answering."""
    
    # Authentication
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication required")
    
    provided_token = auth_header.split(" ")[1]
    if provided_token != Config.BEARER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid Bearer token")
    
    try:
        # Process document
        print("Fetching document...")
        document_text = await fetch_document(request_body.documents)
        
        if not document_text or len(document_text.strip()) < 10:
            raise HTTPException(400, "Could not extract meaningful content from document")
        
        print(f"Document fetched, length: {len(document_text)} characters")
        
        # Process document for RAG (chunk, embed, store)
        print("Processing document for RAG...")
        document_hash = await process_document_for_rag(document_text)
        print(f"Document processed with hash: {document_hash}")
        
        # Process questions and generate answers using RAG
        answers = []
        
        for i, question in enumerate(request_body.questions):
            print(f"Processing question {i+1}/{len(request_body.questions)}: {question}")
            answer = await generate_rag_answer(question, document_hash)
            answers.append(answer)
            print(f"Answer {i+1}: {answer}")
        
        return RunResponse(answers=answers)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Processing failed: {str(e)}")
        raise HTTPException(500, f"Processing failed: {str(e)}")

# For Vercel
app = app
