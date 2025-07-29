import json
import httpx
import asyncio
import os
import io
import logging
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import PyPDF2
from docx import Document
import email
from email.mime.text import MIMEText
import hashlib
import time
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LLM-Powered Intelligent Query-Retrieval System",
    description="Advanced API for processing natural language queries against documents with semantic search and explainable AI"
)

# Global variables for embeddings and vector store
embedding_model = None
vector_store = None
document_chunks = []
chunk_metadata = []

# --- Initialization Function ---
async def initialize_embedding_model():
    """Initialize the embedding model and vector store."""
    global embedding_model
    if embedding_model is None:
        logger.info("Loading embedding model...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Embedding model loaded successfully")

# --- Document Processing Functions ---
class DocumentProcessor:
    @staticmethod
    async def extract_text_from_pdf(content: bytes) -> str:
        """Extract text from PDF content."""
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            return ""

    @staticmethod
    async def extract_text_from_docx(content: bytes) -> str:
        """Extract text from DOCX content."""
        try:
            doc = Document(io.BytesIO(content))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting DOCX text: {e}")
            return ""

    @staticmethod
    async def extract_text_from_email(content: str) -> str:
        """Extract text from email content."""
        try:
            msg = email.message_from_string(content)
            text = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        text += part.get_payload(decode=True).decode('utf-8', errors='ignore')
            else:
                text = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
            return text
        except Exception as e:
            logger.error(f"Error extracting email text: {e}")
            return ""

# --- Text Chunking and Embedding Functions ---
class TextChunker:
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks for better context preservation."""
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append({
                'text': chunk_text,
                'start_index': i,
                'end_index': min(i + chunk_size, len(words)),
                'chunk_id': len(chunks)
            })
            
            if i + chunk_size >= len(words):
                break
                
        return chunks

# --- Vector Store Management ---
class VectorStore:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.index = None
        self.chunks = []
        self.metadata = []
        
    async def build_index(self, chunks: List[Dict[str, Any]], document_url: str):
        """Build FAISS index from document chunks."""
        logger.info("Building vector index...")
        
        # Extract text from chunks
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        # Store chunks and metadata
        self.chunks = chunks
        self.metadata = [{
            'chunk_id': chunk['chunk_id'],
            'start_index': chunk['start_index'],
            'end_index': chunk['end_index'],
            'document_url': document_url,
            'text_length': len(chunk['text'])
        } for chunk in chunks]
        
        logger.info(f"Vector index built with {len(chunks)} chunks")
        
    async def search_similar_chunks(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar chunks using semantic similarity."""
        if self.index is None:
            return []
            
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search in index
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1:  # Valid result
                results.append({
                    'text': self.chunks[idx]['text'],
                    'score': float(score),
                    'metadata': self.metadata[idx],
                    'rank': i + 1
                })
                
        return results

# --- Document Content Fetching ---
async def fetch_document_content(url: str) -> tuple[bytes, str]:
    """Fetch document content and determine file type."""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '').lower()
            content = response.content
            
            # Determine file type
            if 'pdf' in content_type or url.lower().endswith('.pdf'):
                file_type = 'pdf'
            elif 'word' in content_type or url.lower().endswith(('.doc', '.docx')):
                file_type = 'docx'
            elif 'email' in content_type or url.lower().endswith('.eml'):
                file_type = 'email'
            else:
                file_type = 'text'
                
            return content, file_type
            
    except Exception as e:
        logger.error(f"Error fetching document: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching document: {e}")

async def process_document(url: str) -> str:
    """Process document and extract text based on file type."""
    content, file_type = await fetch_document_content(url)
    
    processor = DocumentProcessor()
    
    if file_type == 'pdf':
        return await processor.extract_text_from_pdf(content)
    elif file_type == 'docx':
        return await processor.extract_text_from_docx(content)
    elif file_type == 'email':
        return await processor.extract_text_from_email(content.decode('utf-8', errors='ignore'))
    else:
        return content.decode('utf-8', errors='ignore')

# --- Request and Response Models ---
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

# --- LLM Processing Functions ---
class LLMProcessor:
    @staticmethod
    async def generate_contextual_answer(
        question: str, 
        relevant_chunks: List[Dict[str, Any]], 
        original_document: str = ""
    ) -> AnswerResponse:
        """Generate answer using LLM with retrieved context."""
        
        # Prepare context from relevant chunks
        context = "\n\n".join([
            f"[Chunk {chunk['rank']} - Relevance: {chunk['score']:.3f}]\n{chunk['text']}"
            for chunk in relevant_chunks
        ])
        
        prompt = f"""
You are an expert document analysis assistant. Answer the question based on the provided context chunks from the document.

CONTEXT FROM DOCUMENT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Provide a clear, accurate answer based ONLY on the provided context
2. If the context doesn't contain enough information, state this clearly
3. Include specific details and conditions when available
4. Cite relevant parts of the context to support your answer

ANSWER:
"""

        try:
            # Use OpenAI GPT for better quality responses
            api_key = os.getenv("OPENAI_API_KEY", "")
            if not api_key:
                # Fallback to simpler processing if no API key
                return await LLMProcessor._fallback_answer_generation(question, relevant_chunks)
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "gpt-4-turbo-preview",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 1000,
                        "temperature": 0.1
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    answer = result["choices"][0]["message"]["content"]
                    
                    return AnswerResponse(
                        answer=answer,
                        confidence_score=LLMProcessor._calculate_confidence_score(relevant_chunks),
                        relevant_chunks=relevant_chunks,
                        reasoning=f"Answer derived from {len(relevant_chunks)} relevant document chunks with semantic similarity scores ranging from {min(c['score'] for c in relevant_chunks):.3f} to {max(c['score'] for c in relevant_chunks):.3f}"
                    )
                else:
                    return await LLMProcessor._fallback_answer_generation(question, relevant_chunks)
                    
        except Exception as e:
            logger.error(f"Error in LLM processing: {e}")
            return await LLMProcessor._fallback_answer_generation(question, relevant_chunks)
    
    @staticmethod
    async def _fallback_answer_generation(question: str, relevant_chunks: List[Dict[str, Any]]) -> AnswerResponse:
        """Fallback answer generation using simple text matching."""
        if not relevant_chunks:
            return AnswerResponse(
                answer="Information not available in the provided document.",
                confidence_score=0.0,
                relevant_chunks=[],
                reasoning="No relevant chunks found for the question."
            )
        
        # Simple answer from most relevant chunk
        best_chunk = relevant_chunks[0]
        answer = f"Based on the document: {best_chunk['text'][:500]}..."
        
        return AnswerResponse(
            answer=answer,
            confidence_score=best_chunk['score'],
            relevant_chunks=relevant_chunks,
            reasoning=f"Answer extracted from most relevant chunk (similarity score: {best_chunk['score']:.3f})"
        )
    
    @staticmethod
    def _calculate_confidence_score(chunks: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on chunk relevance."""
        if not chunks:
            return 0.0
        
        # Use highest similarity score as base confidence
        max_score = max(chunk['score'] for chunk in chunks)
        
        # Adjust based on number of relevant chunks
        chunk_bonus = min(0.1 * len(chunks), 0.3)
        
        return min(max_score + chunk_bonus, 1.0)


# --- API Endpoints ---
@app.on_event("startup")
async def startup_event():
    """Initialize the system on startup."""
    await initialize_embedding_model()

@app.post("/hackrx/run", response_model=RunResponse)
async def run_submissions(request_body: RunRequest, request: Request):
    """
    Enhanced endpoint that processes natural language questions against documents
    with semantic search, embeddings, and explainable AI responses.
    """
    start_time = time.time()
    
    # --- Authentication ---
    auth_header = request.headers.get("Authorization")
    expected_token = "ca6914a6c8df9d1ce075149c3ab9f060e666c75940576e37a98b3cf0e9092c72"
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication required: Bearer token missing.")
    
    provided_token = auth_header.split(" ")[1]
    if provided_token != expected_token:
        raise HTTPException(status_code=403, detail="Invalid Bearer token.")

    try:
        # --- Document Processing Pipeline ---
        logger.info(f"Processing document from URL: {request_body.documents}")
        
        # Step 1: Fetch and extract document content
        document_text = await process_document(request_body.documents)
        if not document_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from document")
        
        # Step 2: Chunk the document
        chunker = TextChunker()
        chunks = chunker.chunk_text(document_text)
        logger.info(f"Document chunked into {len(chunks)} pieces")
        
        # Step 3: Build vector index
        vector_store = VectorStore(embedding_model)
        await vector_store.build_index(chunks, request_body.documents)
        
        # Step 4: Process questions with semantic search
        answers = []
        detailed_responses = []
        llm_processor = LLMProcessor()
        
        for question in request_body.questions:
            # Semantic search for relevant chunks
            relevant_chunks = await vector_store.search_similar_chunks(question, k=3)
            
            # Generate contextual answer with explanation
            detailed_response = await llm_processor.generate_contextual_answer(
                question, relevant_chunks, document_text
            )
            
            answers.append(detailed_response.answer)
            detailed_responses.append(detailed_response)
        
        # --- Processing Statistics ---
        processing_time = time.time() - start_time
        stats = {
            "processing_time_seconds": round(processing_time, 2),
            "document_length": len(document_text),
            "total_chunks": len(chunks),
            "questions_processed": len(request_body.questions),
            "average_chunk_size": sum(len(chunk['text']) for chunk in chunks) // len(chunks) if chunks else 0,
            "embedding_model": "all-MiniLM-L6-v2",
            "vector_store": "FAISS",
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

# --- Health Check Endpoint ---
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "embedding_model_loaded": embedding_model is not None,
        "version": "1.0.0"
    }

# --- System Info Endpoint ---
@app.get("/system/info")
async def system_info():
    """Get system information and capabilities."""
    return {
        "system_name": "LLM-Powered Intelligent Query-Retrieval System",
        "capabilities": [
            "PDF document processing",
            "DOCX document processing", 
            "Email document processing",
            "Semantic search with FAISS",
            "Contextual question answering",
            "Explainable AI responses",
            "Token-efficient processing",
            "Real-time performance optimization"
        ],
        "supported_formats": ["PDF", "DOCX", "Email (.eml)", "Plain Text"],
        "embedding_model": "all-MiniLM-L6-v2",
        "vector_database": "FAISS",
        "chunking_strategy": "Overlapping windows with 1000 tokens, 200 overlap",
        "max_relevant_chunks": 3,
        "processing_pipeline": [
            "Document fetching and type detection",
            "Content extraction based on format",
            "Text chunking with overlap",
            "Embedding generation", 
            "Vector index creation",
            "Semantic similarity search",
            "Contextual answer generation",
            "Explainability and confidence scoring"
        ]
    }
