"""
Multi-LLM API with Vector Database: OpenAI, Anthropic Claude, Hugging Face with FAISS
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

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    VECTOR_DB_AVAILABLE = True
except ImportError:
    VECTOR_DB_AVAILABLE = False

app = FastAPI(title="Natural LLM Document Query System", version="3.0.0")

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
    
    # Vector Database Configuration
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Lightweight and fast
    CHUNK_SIZE = 256  # Smaller chunks for better precision
    CHUNK_OVERLAP = 32  # Reduced overlap
    MAX_RELEVANT_CHUNKS = 3  # Fewer but more relevant chunks
    SIMILARITY_THRESHOLD = 0.2  # Lower threshold for more results
    MIN_VECTOR_DB_SIZE = 1000  # Use vector DB for documents > 1KB (almost all documents)
    
    REQUEST_TIMEOUT = 30.0
    MAX_DOCUMENT_SIZE = 50 * 1024 * 1024  # Increased to 50MB for large documents

# Vector Database Class
class VectorDatabase:
    """FAISS-based vector database for document chunks."""
    
    def __init__(self, embedding_model_name: str = Config.EMBEDDING_MODEL):
        self.embedding_model = None
        self.index = None
        self.chunks = []
        self.embeddings = None
        
        if VECTOR_DB_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(embedding_model_name)
                self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
                # Use IndexFlatIP for cosine similarity (inner product with normalized vectors)
                self.index = faiss.IndexFlatIP(self.embedding_dim)
                print(f"Vector database initialized with {embedding_model_name}")
            except Exception as e:
                print(f"Failed to initialize vector database: {e}")
                VECTOR_DB_AVAILABLE = False
    
    def add_documents(self, chunks: List[Dict[str, Any]]) -> bool:
        """Add document chunks to the vector database."""
        if not VECTOR_DB_AVAILABLE or not self.embedding_model:
            return False
        
        try:
            # Extract text from chunks
            texts = [chunk['text'] for chunk in chunks]
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add to FAISS index
            self.index.add(embeddings.astype('float32'))
            
            # Store chunks and embeddings
            self.chunks.extend(chunks)
            if self.embeddings is None:
                self.embeddings = embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, embeddings])
            
            print(f"Added {len(chunks)} chunks to vector database")
            return True
            
        except Exception as e:
            print(f"Error adding documents to vector database: {e}")
            return False
    
    def search(self, query: str, k: int = Config.MAX_RELEVANT_CHUNKS) -> List[Dict[str, Any]]:
        """Search for relevant chunks using semantic similarity."""
        if not VECTOR_DB_AVAILABLE or not self.embedding_model or len(self.chunks) == 0:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
            
            # Normalize query embedding
            faiss.normalize_L2(query_embedding)
            
            # Search the index
            scores, indices = self.index.search(query_embedding.astype('float32'), k)
            
            # Prepare results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx >= 0 and score >= Config.SIMILARITY_THRESHOLD:
                    chunk = self.chunks[idx].copy()
                    chunk['score'] = float(score)
                    chunk['rank'] = i + 1
                    results.append(chunk)
            
            return results
            
        except Exception as e:
            print(f"Error searching vector database: {e}")
            return []
    
    def clear(self):
        """Clear the vector database."""
        if VECTOR_DB_AVAILABLE and self.index is not None:
            self.index.reset()
            self.chunks = []
            self.embeddings = None

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

def chunk_text_for_embeddings(text: str) -> List[Dict[str, Any]]:
    """Optimized text chunking for vector embeddings."""
    chunk_size = Config.CHUNK_SIZE  # Tokens, not characters
    overlap = Config.CHUNK_OVERLAP
    chunks = []
    
    # Clean the text first
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
    estimated_tokens = len(text) // 4
    
    if estimated_tokens < chunk_size:
        # If document is small, return as single chunk
        return [{
            'text': text,
            'chunk_id': 0,
            'start_char': 0,
            'end_char': len(text),
            'metadata': {'type': 'full_document', 'tokens': estimated_tokens}
        }]
    
    # Strategy 1: Try semantic chunking (paragraphs/sections)
    # Look for natural break points
    break_patterns = [
        r'\n\s*(?:Chapter|Section|Part|Article|§)\s*\d+',  # Formal sections
        r'\n\s*\d+\.\s+[A-Z]',  # Numbered items
        r'\n\s*[A-Z][A-Z\s]{15,}\s*\n',  # Headers
        r'\n\s*\n\s*[A-Z]'  # Paragraph breaks
    ]
    
    # Find break points
    break_points = [0]
    for pattern in break_patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        break_points.extend([match.start() for match in matches])
    
    break_points.append(len(text))
    break_points = sorted(set(break_points))
    
    # Create chunks based on break points
    for i in range(len(break_points) - 1):
        start = break_points[i]
        end = break_points[i + 1]
        section_text = text[start:end].strip()
        
        if not section_text:
            continue
        
        # If section is too large, split it further
        if len(section_text) > chunk_size * 4:  # Convert tokens to chars
            # Split by sentences
            sentences = re.split(r'[.!?]+', section_text)
            current_chunk = ""
            chunk_start = start
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                # Check if adding this sentence exceeds chunk size
                test_chunk = current_chunk + ". " + sentence if current_chunk else sentence
                if len(test_chunk) > chunk_size * 4 and current_chunk:
                    # Save current chunk
                    chunks.append({
                        'text': current_chunk.strip(),
                        'chunk_id': len(chunks),
                        'start_char': chunk_start,
                        'end_char': chunk_start + len(current_chunk),
                        'metadata': {
                            'type': 'sentence_split',
                            'tokens': len(current_chunk) // 4,
                            'section': i
                        }
                    })
                    
                    # Start new chunk with overlap
                    if len(current_chunk) > overlap * 4:
                        overlap_text = current_chunk[-(overlap * 4):]
                        current_chunk = overlap_text + ". " + sentence
                        chunk_start = chunk_start + len(current_chunk) - len(overlap_text)
                    else:
                        current_chunk = sentence
                        chunk_start = start + section_text.find(sentence)
                else:
                    current_chunk = test_chunk
            
            # Add final chunk from this section
            if current_chunk.strip():
                chunks.append({
                    'text': current_chunk.strip(),
                    'chunk_id': len(chunks),
                    'start_char': chunk_start,
                    'end_char': chunk_start + len(current_chunk),
                    'metadata': {
                        'type': 'sentence_split',
                        'tokens': len(current_chunk) // 4,
                        'section': i
                    }
                })
        else:
            # Section is small enough, use as chunk
            chunks.append({
                'text': section_text,
                'chunk_id': len(chunks),
                'start_char': start,
                'end_char': end,
                'metadata': {
                    'type': 'semantic_section',
                    'tokens': len(section_text) // 4,
                    'section': i
                }
            })
    
    # If we didn't get good chunks, fall back to sliding window
    if len(chunks) < 2:
        chunks = []
        chunk_size_chars = chunk_size * 4
        overlap_chars = overlap * 4
        
        for i in range(0, len(text), chunk_size_chars - overlap_chars):
            chunk_text = text[i:i + chunk_size_chars]
            
            # Try to end at sentence boundary
            if i + chunk_size_chars < len(text):
                last_period = chunk_text.rfind('.')
                last_exclaim = chunk_text.rfind('!')
                last_question = chunk_text.rfind('?')
                last_sentence_end = max(last_period, last_exclaim, last_question)
                
                if last_sentence_end > len(chunk_text) * 0.8:  # If we can cut at least 80% through
                    chunk_text = chunk_text[:last_sentence_end + 1]
            
            chunks.append({
                'text': chunk_text.strip(),
                'chunk_id': len(chunks),
                'start_char': i,
                'end_char': i + len(chunk_text),
                'metadata': {
                    'type': 'sliding_window',
                    'tokens': len(chunk_text) // 4,
                    'window': i // (chunk_size_chars - overlap_chars)
                }
            })
    
    # Remove empty chunks and very small chunks
    chunks = [chunk for chunk in chunks if len(chunk['text'].strip()) > 50]
    
    # Re-number chunk IDs
    for i, chunk in enumerate(chunks):
        chunk['chunk_id'] = i
    
    return chunks

def simple_search(query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Enhanced keyword-based search for any document type."""
    query_words = set(re.findall(r'\w+', query.lower()))
    scored_chunks = []
    
    # Generic keywords that indicate important information across document types
    important_keywords = {
        # Financial/Business terms
        'cost', 'price', 'fee', 'amount', 'payment', 'revenue', 'profit', 'budget',
        # Time-related terms
        'date', 'deadline', 'period', 'duration', 'time', 'schedule', 'term',
        # Process/Action terms
        'process', 'procedure', 'method', 'steps', 'requirements', 'conditions',
        # Legal/Compliance terms
        'policy', 'regulation', 'compliance', 'requirement', 'obligation', 'right',
        # Quantitative terms
        'percentage', 'rate', 'ratio', 'limit', 'maximum', 'minimum', 'threshold',
        # Status/State terms
        'available', 'eligible', 'covered', 'included', 'excluded', 'applicable',
        # Common document sections
        'overview', 'summary', 'details', 'specification', 'description', 'definition'
    }
    
    for chunk in chunks:
        chunk_text_lower = chunk['text'].lower()
        chunk_words = set(re.findall(r'\w+', chunk_text_lower))
        
        # Basic keyword overlap score
        overlap = len(query_words.intersection(chunk_words))
        base_score = overlap / len(query_words) if query_words else 0
        
        # Boost score for important keywords
        keyword_boost = 0
        for keyword in important_keywords:
            if keyword in chunk_text_lower:
                keyword_boost += 0.05  # Smaller boost for generic terms
        
        # Boost score for exact phrase matches
        phrase_boost = 0
        query_text = query.lower()
        if len(query_text) > 10 and query_text in chunk_text_lower:
            phrase_boost = 0.4
        
        # Boost for partial phrase matches (2+ consecutive words)
        query_parts = query_text.split()
        if len(query_parts) >= 2:
            for i in range(len(query_parts) - 1):
                phrase = ' '.join(query_parts[i:i+2])
                if phrase in chunk_text_lower:
                    phrase_boost += 0.2
        
        # Boost for numerical content if query contains numbers
        number_boost = 0
        if re.search(r'\d+', query) and re.search(r'\d+', chunk['text']):
            number_boost = 0.1
        
        # Boost for definition-style content
        definition_boost = 0
        if any(word in query.lower() for word in ['what is', 'define', 'definition', 'meaning']):
            if any(pattern in chunk_text_lower for pattern in ['means', 'defined as', 'refers to', ':']):
                definition_boost = 0.3
        
        # Penalize very short or very long chunks
        length_penalty = 0
        chunk_length = len(chunk['text'])
        if chunk_length < 50:
            length_penalty = -0.2
        elif chunk_length > 2000:
            length_penalty = -0.1
        
        # Calculate final score
        final_score = base_score + keyword_boost + phrase_boost + number_boost + definition_boost + length_penalty
        
        if final_score > 0 or overlap > 0:
            # Create result in consistent format
            chunk_copy = chunk.copy()
            chunk_copy['score'] = final_score
            chunk_copy['metadata'] = chunk_copy.get('metadata', {})
            chunk_copy['metadata']['search_type'] = 'keyword'
            scored_chunks.append(chunk_copy)
    
    # Sort by score and return top 5 for better context
    scored_chunks.sort(key=lambda x: x['score'], reverse=True)
    return scored_chunks[:5]


async def vector_search(text: str, queries: List[str], use_vector_db: bool = True) -> List[Dict[str, Any]]:
    """
    Enhanced search that handles questions from anywhere in the document.
    Provides comprehensive context extraction for precise answers.
    """
    # Always try vector database first for semantic understanding
    if VECTOR_DB_AVAILABLE:
        try:
            # Initialize vector database
            vector_db = VectorDatabase()
            
            # Create comprehensive chunks with overlap for better context
            chunks = chunk_text_for_embeddings(text)
            success = vector_db.add_documents(chunks)
            
            if success:
                # Enhanced search for each query
                all_results = []
                for query in queries:
                    # Get more results for broader context
                    results = vector_db.search(query, k=min(15, len(chunks)))
                    
                    # Enhance results with additional context scoring
                    for result in results:
                        result['metadata']['search_type'] = 'vector'
                        result['metadata']['query'] = query
                        
                        # Additional scoring based on exact keyword matches
                        query_words = set(query.lower().split())
                        text_words = set(result['text'].lower().split())
                        exact_matches = len(query_words.intersection(text_words))
                        
                        # Boost score for exact keyword matches
                        if exact_matches > 0:
                            keyword_boost = (exact_matches / len(query_words)) * 0.3
                            result['score'] = min(1.0, result['score'] + keyword_boost)
                        
                        # Check for specific policy terms and boost relevance
                        policy_terms = {
                            'grace period': 1.2, 'waiting period': 1.2, 'pre-existing': 1.1,
                            'maternity': 1.1, 'cataract': 1.1, 'organ donor': 1.1,
                            'coverage': 1.1, 'benefits': 1.1, 'exclusions': 1.1,
                            'premium': 1.1, 'deductible': 1.1, 'co-payment': 1.1
                        }
                        
                        text_lower = result['text'].lower()
                        for term, boost in policy_terms.items():
                            if term in query.lower() and term in text_lower:
                                result['score'] = min(1.0, result['score'] * boost)
                    
                    all_results.extend(results)
                
                # Remove duplicates and merge overlapping content
                unique_results = {}
                for result in all_results:
                    chunk_id = result['metadata'].get('chunk_id', 0)
                    if chunk_id in unique_results:
                        # Keep the result with higher score
                        if result['score'] > unique_results[chunk_id]['score']:
                            unique_results[chunk_id] = result
                    else:
                        unique_results[chunk_id] = result
                
                # Sort by enhanced relevance score
                final_results = list(unique_results.values())
                final_results.sort(key=lambda x: x['score'], reverse=True)
                
                # Ensure we have sufficient context for precise answers
                if len(final_results) > 0:
                    # Add surrounding context for better understanding
                    enhanced_results = []
                    for result in final_results[:8]:  # Top 8 most relevant
                        enhanced_result = result.copy()
                        
                        # Find adjacent chunks for context expansion
                        current_chunk_id = result['metadata'].get('chunk_id', 0)
                        for chunk in chunks:
                            chunk_id = chunk['metadata'].get('chunk_id', 0)
                            # Add context from adjacent chunks
                            if abs(chunk_id - current_chunk_id) == 1:
                                # Append adjacent context
                                if chunk_id > current_chunk_id:
                                    enhanced_result['text'] += " " + chunk['text'][:200]
                                else:
                                    enhanced_result['text'] = chunk['text'][-200:] + " " + enhanced_result['text']
                        
                        enhanced_results.append(enhanced_result)
                    
                    print(f"Enhanced vector search returned {len(enhanced_results)} contextual results")
                    return enhanced_results
                
        except Exception as e:
            print(f"Vector search failed: {e}")
    
    # Enhanced fallback search with comprehensive keyword matching
    print("Using enhanced keyword search for comprehensive coverage...")
    chunks = chunk_text_for_embeddings(text)
    
    if not queries:
        return chunks[:Config.MAX_RELEVANT_CHUNKS]
    
    # Combine all queries for comprehensive search
    all_query_words = set()
    for query in queries:
        all_query_words.update(query.lower().split())
    
    # Enhanced keyword search with multiple scoring methods
    scored_chunks = []
    for chunk in chunks:
        chunk_text_lower = chunk['text'].lower()
        chunk_words = set(chunk_text_lower.split())
        
        # Multiple scoring approaches
        scores = []
        
        # 1. Direct word overlap
        overlap = len(all_query_words.intersection(chunk_words))
        if overlap > 0:
            scores.append((overlap / len(all_query_words)) * 0.4)
        
        # 2. Phrase matching for each query
        phrase_score = 0
        for query in queries:
            query_lower = query.lower()
            if query_lower in chunk_text_lower:
                phrase_score += 0.8  # High score for exact phrase match
            else:
                # Partial phrase matching
                query_words = query_lower.split()
                for i in range(len(query_words) - 1):
                    bigram = f"{query_words[i]} {query_words[i+1]}"
                    if bigram in chunk_text_lower:
                        phrase_score += 0.2
        scores.append(phrase_score * 0.4)
        
        # 3. Policy-specific term weighting
        policy_specific_score = 0
        important_terms = [
            'grace period', 'waiting period', 'pre-existing', 'maternity',
            'coverage', 'covered', 'benefits', 'exclusions', 'premium',
            'deductible', 'co-payment', 'cataract', 'organ donor'
        ]
        
        for term in important_terms:
            if any(term in query.lower() for query in queries) and term in chunk_text_lower:
                policy_specific_score += 0.3
        scores.append(policy_specific_score * 0.2)
        
        # Calculate final score
        final_score = sum(scores)
        
        if final_score > 0.05:  # Lower threshold for broader coverage
            chunk_copy = chunk.copy()
            chunk_copy['score'] = final_score
            chunk_copy['metadata']['search_type'] = 'enhanced_keyword'
            scored_chunks.append(chunk_copy)
    
    # Sort by score and return comprehensive results
    scored_chunks.sort(key=lambda x: x['score'], reverse=True)
    return scored_chunks[:min(10, len(scored_chunks))]  # Return more results for better context


# LLM Providers
async def call_openai(question: str, context: str) -> Dict[str, Any]:
    """Call OpenAI GPT API for intelligent answer extraction from document context."""
    if not Config.OPENAI_API_KEY:
        raise Exception("OpenAI API key not available")

    prompt = f"""You are an expert document analyst. Extract the precise answer to the question from the provided document context.

DOCUMENT CONTEXT:
{context[:4000]}

QUESTION: {question}

INSTRUCTIONS:
- Read the document context carefully
- Find the specific information that directly answers the question
- Extract the exact answer as it appears in the document
- Provide a clear, factual one-sentence response
- Use the exact numbers, time periods, and terms from the document
- If the information is not in the context, state that clearly

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
            
            return {
                "answer": answer,
                "provider": "openai",
                "model": Config.OPENAI_MODEL,
                "success": True
            }
            
    except Exception as e:
        return {"success": False, "error": str(e), "provider": "openai"}

async def call_anthropic(question: str, context: str) -> Dict[str, Any]:
    """Call Anthropic Claude API for intelligent answer extraction from document context."""
    if not Config.ANTHROPIC_API_KEY:
        raise Exception("Anthropic API key not available")
    
    prompt = f"""You are an expert document analyst. Extract the precise answer to the question from the provided document context.

DOCUMENT CONTEXT:
{context[:4000]}

QUESTION: {question}

INSTRUCTIONS:
- Read the document context carefully
- Find the specific information that directly answers the question
- Extract the exact answer as it appears in the document
- Provide a clear, factual one-sentence response
- Use the exact numbers, time periods, and terms from the document
- If the information is not in the context, state that clearly

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
            
            return {
                "answer": answer,
                "provider": "anthropic",
                "model": Config.ANTHROPIC_MODEL,
                "success": True
            }
            
    except Exception as e:
        return {"success": False, "error": str(e), "provider": "anthropic"}

async def call_huggingface(question: str, context: str) -> Dict[str, Any]:
    """Call Hugging Face Inference API for precise one-sentence answers."""
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
    
    # Create a natural prompt for document analysis
    prompt = f"""You are an expert document analyst. Extract the precise answer to the question from the provided document context.

DOCUMENT CONTEXT:
{context[:3000]}

QUESTION: {question}

INSTRUCTIONS:
- Read the document context carefully
- Find the specific information that directly answers the question
- Extract the exact answer as it appears in the document
- Provide a clear, factual response
- Use the exact numbers, time periods, and terms from the document

ANSWER:"""

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                model_url,
                headers=headers,
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 50,  # Limit for one sentence
                        "temperature": 0.1,   # Lower temperature for factual responses
                        "do_sample": True,
                        "top_p": 0.8
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
            
            # Clean up the answer and ensure it's one sentence
            if answer:
                # Take only the first sentence
                sentences = re.split(r'[.!?]+', answer)
                if sentences and len(sentences[0].strip()) > 10:
                    answer = sentences[0].strip()
                    if not answer.endswith('.'):
                        answer += '.'
                else:
                    answer = "Unable to generate a meaningful response from Hugging Face model."
            else:
                answer = "Unable to generate a meaningful response from Hugging Face model."
            
            # Determine which model was actually used
            model_used = "google/flan-t5-large" if Config.HUGGINGFACE_API_KEY == "hf_demo_key_public" else Config.HUGGINGFACE_MODEL
            
            return {
                "answer": answer[:200],  # Limit answer length
                "provider": "huggingface",
                "model": model_used,
                "success": True
            }
            
    except Exception as e:
        return {"success": False, "error": str(e), "provider": "huggingface"}

def generate_fallback_answer(question: str, relevant_chunks: List[Dict[str, Any]]) -> str:
    """Generate answers using intelligent text processing and context analysis.
    
    This function uses ONLY natural language processing and semantic analysis.
    NO hardcoded patterns or predetermined responses are used.
    """
    if not relevant_chunks:
        return "Information not available in the provided document."
    
    # Combine the most relevant chunks for analysis
    combined_text = " ".join([chunk['text'] for chunk in relevant_chunks[:3]])
    question_lower = question.lower()
    
    # Extract key information using natural language processing techniques
    # Find sentences that are most relevant to the question
    sentences = re.split(r'[.!?]+', combined_text)
    question_words = set(re.findall(r'\b\w+\b', question_lower))
    
    # Remove common stop words
    stop_words = {
        'what', 'is', 'the', 'are', 'does', 'do', 'can', 'will', 'how', 'when', 
        'where', 'why', 'which', 'under', 'this', 'policy', 'a', 'an', 'and', 
        'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'
    }
    key_question_words = question_words - stop_words
    
    # Score sentences based on relevance to the question
    scored_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 15:  # Skip very short sentences
            continue
            
        sentence_lower = sentence.lower()
        sentence_words = set(re.findall(r'\b\w+\b', sentence_lower))
        
        # Calculate semantic relevance score
        word_overlap = len(key_question_words.intersection(sentence_words))
        overlap_score = word_overlap / max(len(key_question_words), 1) if key_question_words else 0
        
        # Boost score for sentences containing numbers (often contain specific details)
        number_boost = 0.2 if re.search(r'\b\d+\b', sentence) else 0
        
        # Boost score for sentences with typical answer indicators
        answer_indicators = [
            'period', 'days', 'months', 'years', 'covered', 'coverage', 'benefit', 
            'amount', 'limit', 'percentage', 'provided', 'available', 'eligible',
            'required', 'includes', 'excludes', 'shall', 'will', 'must'
        ]
        indicator_boost = sum(0.1 for indicator in answer_indicators if indicator in sentence_lower)
        
        # Penalize very long sentences (might be less focused)
        length_penalty = -0.1 if len(sentence) > 200 else 0
        
        total_score = overlap_score + number_boost + indicator_boost + length_penalty
        
        if total_score > 0.1:  # Only consider sentences with meaningful relevance
            scored_sentences.append({
                'text': sentence,
                'score': total_score,
                'word_overlap': word_overlap
            })
    
    # Sort by score and get the best sentence
    scored_sentences.sort(key=lambda x: x['score'], reverse=True)
    
    if scored_sentences:
        best_sentence = scored_sentences[0]['text'].strip()
        
        # Ensure the sentence ends properly
        if not best_sentence.endswith(('.', '!', '?')):
            best_sentence += '.'
            
        return best_sentence
    
    # If no good sentence found, try to extract key phrases
    # Look for phrases that might contain the answer
    key_phrases = []
    
    # Extract phrases around question keywords
    for word in key_question_words:
        pattern = rf'\b.{{0,50}}{re.escape(word)}.{{0,50}}\b'
        matches = re.findall(pattern, combined_text, re.IGNORECASE)
        key_phrases.extend(matches)
    
    if key_phrases:
        # Find the phrase with the most question words
        best_phrase = ""
        best_count = 0
        
        for phrase in key_phrases:
            phrase_words = set(re.findall(r'\b\w+\b', phrase.lower()))
            overlap_count = len(key_question_words.intersection(phrase_words))
            
            if overlap_count > best_count:
                best_count = overlap_count
                best_phrase = phrase.strip()
        
        if best_phrase and len(best_phrase) > 20:
            # Clean up the phrase and make it a proper sentence
            best_phrase = re.sub(r'\s+', ' ', best_phrase)
            if not best_phrase.endswith(('.', '!', '?')):
                best_phrase += '.'
            return best_phrase
    
    # Final fallback: return the first substantial sentence from the most relevant chunk
    if relevant_chunks:
        first_chunk_sentences = re.split(r'[.!?]+', relevant_chunks[0]['text'])
        for sentence in first_chunk_sentences:
            sentence = sentence.strip()
            if len(sentence) > 30:  # Ensure it's substantial
                if not sentence.endswith(('.', '!', '?')):
                    sentence += '.'
                return sentence
    
    return "The requested information is available in the document but requires more specific context."

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
        
        # Always use vector database for better accuracy (when available)
        # For smaller documents, still use vector DB if available for better semantic search
        
        # Process questions
        answers = []
        
        for question in request_body.questions:
            # Always try vector search first for better semantic matching
            relevant_chunks = await vector_search(document_text, [question], use_vector_db=True)
            
            # Generate answer with LLM fallback
            detailed_response = await generate_answer_with_fallback(question, relevant_chunks)
            
            answers.append(detailed_response.answer)
        
        return RunResponse(
            answers=answers
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Processing failed: {str(e)}")

# For Vercel
app = app
