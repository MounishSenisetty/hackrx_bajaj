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
    SIMILARITY_THRESHOLD = 0.5  # Lowered from 0.7 to find more relevant content
    MAX_RELEVANT_CHUNKS = 5  # Increased from 3 to get more context
    ENABLE_GENERAL_KNOWLEDGE = True  # Allow LLM to use general knowledge





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
    """Enhanced PDF text extraction with multiple fallback methods"""
    try:
        pdf_file = io.BytesIO(content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        # Method 1: Standard text extraction
        full_text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    full_text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
            except Exception as e:
                print(f"Failed to extract text from page {page_num + 1}: {e}")
                continue
        
        # Clean up the extracted text
        if full_text.strip():
            # Remove excessive whitespace and normalize
            full_text = re.sub(r'\s+', ' ', full_text)
            full_text = re.sub(r'\n\s*\n', '\n', full_text)
            full_text = full_text.strip()
            
            # Check if we got meaningful content (not just metadata)
            meaningful_words = len(re.findall(r'\b[a-zA-Z]{3,}\b', full_text))
            if meaningful_words > 10 and len(full_text) > 100:
                print(f"Successfully extracted {len(full_text)} characters from PDF with {meaningful_words} meaningful words")
                return full_text
        
        # Method 2: Try alternative extraction if standard method fails
        print("Standard PDF extraction failed, trying alternative method...")
        pdf_file.seek(0)  # Reset file pointer
        
        try:
            import fitz  # PyMuPDF - better for complex PDFs
            doc = fitz.open(stream=content, filetype="pdf")
            alternative_text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                if page_text.strip():
                    alternative_text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
            
            doc.close()
            
            if alternative_text.strip():
                alternative_text = re.sub(r'\s+', ' ', alternative_text)
                alternative_text = re.sub(r'\n\s*\n', '\n', alternative_text)
                alternative_text = alternative_text.strip()
                
                meaningful_words = len(re.findall(r'\b[a-zA-Z]{3,}\b', alternative_text))
                if meaningful_words > 10:
                    print(f"Alternative PDF extraction successful: {len(alternative_text)} characters")
                    return alternative_text
                    
        except ImportError:
            print("PyMuPDF not available, continuing with PyPDF2 only")
        except Exception as e:
            print(f"Alternative PDF extraction failed: {e}")
        
        # Method 3: Force extract even partial content
        print("Trying to extract any available text from PDF...")
        if full_text.strip():
            return full_text.strip()
        
        # Last resort: return raw text extraction attempt
        return extract_text_fallback(content)
        
    except Exception as e:
        print(f"PDF extraction completely failed: {e}")
        return extract_text_fallback(content)

def extract_text_fallback(content: bytes) -> str:
    """Enhanced fallback text extraction for various file types"""
    try:
        # Method 1: Try UTF-8 decoding
        try:
            text = content.decode('utf-8', errors='ignore')
            text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Check if this looks like meaningful text
            meaningful_words = len(re.findall(r'\b[a-zA-Z]{3,}\b', text))
            if meaningful_words > 5 and len(text) > 50:
                print(f"UTF-8 extraction successful: {len(text)} characters, {meaningful_words} meaningful words")
                return text
        except Exception:
            pass
        
        # Method 2: Try Latin-1 encoding
        try:
            text_latin1 = content.decode('latin-1', errors='ignore')
            text_latin1 = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', ' ', text_latin1)
            text_latin1 = re.sub(r'\s+', ' ', text_latin1).strip()
            
            meaningful_words = len(re.findall(r'\b[a-zA-Z]{3,}\b', text_latin1))
            if meaningful_words > 5 and len(text_latin1) > 50:
                print(f"Latin-1 extraction successful: {len(text_latin1)} characters")
                return text_latin1
        except Exception:
            pass
        
        # Method 3: Try to detect if this is HTML/XML and extract text
        content_str = content.decode('utf-8', errors='ignore')
        if '<html' in content_str.lower() or '<?xml' in content_str.lower():
            try:
                # Simple HTML/XML tag removal
                text = re.sub(r'<[^>]+>', ' ', content_str)
                text = re.sub(r'&[a-zA-Z0-9#]+;', ' ', text)  # Remove HTML entities
                text = re.sub(r'\s+', ' ', text).strip()
                
                meaningful_words = len(re.findall(r'\b[a-zA-Z]{3,}\b', text))
                if meaningful_words > 10:
                    print(f"HTML/XML extraction successful: {len(text)} characters")
                    return text
            except Exception:
                pass
        
        # Method 4: Check if this might be a PDF that failed extraction
        if content.startswith(b'%PDF'):
            print("Detected PDF file, but text extraction failed")
            return "This appears to be a PDF file, but text extraction failed. The content may be scanned images or the PDF may be encrypted."
        
        # Method 5: Look for any readable text in the binary content
        readable_text = ""
        try:
            # Extract sequences of printable ASCII characters
            import string
            current_word = ""
            for byte in content:
                char = chr(byte) if byte < 128 else ''
                if char in string.printable and char not in '\x0b\x0c':
                    current_word += char
                else:
                    if len(current_word) > 3:  # Only keep words longer than 3 chars
                        readable_text += current_word + " "
                    current_word = ""
            
            if len(current_word) > 3:
                readable_text += current_word
                
            readable_text = re.sub(r'\s+', ' ', readable_text).strip()
            meaningful_words = len(re.findall(r'\b[a-zA-Z]{3,}\b', readable_text))
            
            if meaningful_words > 10 and len(readable_text) > 100:
                print(f"Binary text extraction found {meaningful_words} meaningful words")
                return readable_text
                
        except Exception:
            pass
        
        # Last resort: return whatever we can decode
        try:
            fallback_text = content.decode('utf-8', errors='replace')
            fallback_text = re.sub(r'[^\x20-\x7E\n\r\t]', ' ', fallback_text)
            fallback_text = re.sub(r'\s+', ' ', fallback_text).strip()
            
            if len(fallback_text) > 20:
                return fallback_text[:1000] + ("..." if len(fallback_text) > 1000 else "")
        except Exception:
            pass
        
        return "Document content could not be decoded or extracted"
        
    except Exception as e:
        print(f"All text extraction methods failed: {e}")
        return f"Document processing failed: {str(e)}"

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
async def call_openai_api(question: str, context: str, use_general_knowledge: bool = False) -> str:
    """Call OpenAI GPT API with context-aware prompting"""
    if not Config.OPENAI_API_KEY:
        return None
    
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {Config.OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Create context-aware system message
    if use_general_knowledge:
        system_msg = """You are a helpful assistant. Answer the question using the provided context if relevant. 
        If the context doesn't contain enough information to fully answer the question, you may supplement with general knowledge, 
        but clearly indicate when you're doing so. Be specific and helpful."""
    else:
        system_msg = """Answer the question based on the provided context. Be specific and extract relevant details. 
        If the context mentions related concepts but not the exact answer, explain what information is available."""
    
    user_content = f"Context from document: {context}\n\nQuestion: {question}"
    
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content}
        ],
        "max_tokens": 200,
        "temperature": 0.2
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

async def call_google_api(question: str, context: str, use_general_knowledge: bool = False) -> str:
    """Call Google Gemini API with context-aware prompting"""
    if not Config.GOOGLE_API_KEY:
        return None
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={Config.GOOGLE_API_KEY}"
    headers = {"Content-Type": "application/json"}
    
    if use_general_knowledge:
        prompt = f"""Based on the provided context, answer the question. If the context doesn't have complete information, 
        you may use general knowledge to provide a helpful answer, but indicate when you're supplementing with external knowledge.

        Context: {context}
        
        Question: {question}
        
        Provide a comprehensive and helpful answer."""
    else:
        prompt = f"""Based on this context, answer the question. Extract specific details and be thorough:

        Context: {context}
        
        Question: {question}"""
    
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "maxOutputTokens": 200,
            "temperature": 0.2
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

async def call_general_knowledge_api(question: str) -> str:
    """Call LLM APIs to answer using general knowledge when document doesn't have the answer"""
    # Try OpenAI first for general knowledge
    if Config.OPENAI_API_KEY:
        try:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {Config.OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant. Answer the question using your general knowledge. Be informative and accurate."},
                    {"role": "user", "content": f"Question: {question}\n\nNote: The provided document doesn't contain relevant information for this question, so please answer using general knowledge."}
                ],
                "max_tokens": 200,
                "temperature": 0.3
            }
            
            async with httpx.AsyncClient(timeout=8.0) as client:
                response = await client.post(url, headers=headers, json=payload)
                if response.status_code == 200:
                    result = response.json()
                    answer = result["choices"][0]["message"]["content"].strip()
                    return f"Based on general knowledge: {answer}"
        except Exception:
            pass
    
    # Try Google as fallback
    if Config.GOOGLE_API_KEY:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={Config.GOOGLE_API_KEY}"
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [{
                    "parts": [{
                        "text": f"The provided document doesn't contain information about this question. Please answer using general knowledge: {question}"
                    }]
                }],
                "generationConfig": {
                    "maxOutputTokens": 200,
                    "temperature": 0.3
                }
            }
            
            async with httpx.AsyncClient(timeout=8.0) as client:
                response = await client.post(url, headers=headers, json=payload)
                if response.status_code == 200:
                    result = response.json()
                    answer = result["candidates"][0]["content"]["parts"][0]["text"].strip()
                    return f"Based on general knowledge: {answer}"
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

async def generate_answer(question: str, context: str, has_relevant_context: bool = True) -> str:
    """Generate answer with intelligent context handling and general knowledge fallback"""
    
    # If we have relevant context, try context-based answering first
    if has_relevant_context and context.strip():
        # Try OpenAI with context
        answer = await call_openai_api(question, context, use_general_knowledge=False)
        if answer and not is_generic_response(answer):
            return answer
        
        # Try Google with context
        answer = await call_google_api(question, context, use_general_knowledge=False)
        if answer and not is_generic_response(answer):
            return answer
        
        # Try with general knowledge enabled (context + general knowledge)
        if Config.ENABLE_GENERAL_KNOWLEDGE:
            answer = await call_openai_api(question, context, use_general_knowledge=True)
            if answer and not is_generic_response(answer):
                return answer
            
            answer = await call_google_api(question, context, use_general_knowledge=True)
            if answer and not is_generic_response(answer):
                return answer
    
    # If context-based answering fails or no relevant context, try general knowledge
    if Config.ENABLE_GENERAL_KNOWLEDGE:
        general_answer = await call_general_knowledge_api(question)
        if general_answer:
            return general_answer
    
    # Final fallback to rule-based extraction
    if context.strip():
        return extract_answer_from_context(question, context)
    else:
        return "I don't have enough information in the provided document to answer this question."

def is_generic_response(answer: str) -> bool:
    """Check if the response is too generic or indicates lack of information"""
    answer_lower = answer.lower()
    generic_phrases = [
        "does not specify",
        "not mentioned",
        "doesn't mention",
        "not provided",
        "no information",
        "doesn't contain",
        "not available",
        "insufficient information",
        "cannot be determined",
        "not clear",
        "not stated",
        "cannot be answered from",
        "cannot answer",
        "provided text",
        "given context",
        "cannot find",
        "not contain any information",
        "focuses on",
        "therefore",
        "I cannot answer"
    ]
    
    # Check if answer contains generic/inability phrases regardless of length
    if any(phrase in answer_lower for phrase in generic_phrases):
        # Additional check: if it mentions the document topic but says it can't answer the question
        document_indicators = ["visual studio code", "vs code", "microsoft", "github", "contribute"]
        question_rejection_patterns = [
            "about.*insurance",
            "about.*python", 
            "about.*machine learning",
            "about.*artificial intelligence",
            "therefore.*cannot",
            "does.*not.*contain.*information.*about"
        ]
        
        # If it mentions the document but rejects the question topic, it's generic
        has_document_context = any(indicator in answer_lower for indicator in document_indicators)
        rejects_question = any(re.search(pattern, answer_lower) for pattern in question_rejection_patterns)
        
        if has_document_context and rejects_question:
            return True
        
        # Also check for simple rejection patterns
        simple_rejections = [
            "cannot be answered",
            "does not contain",
            "therefore.*cannot",
            "no information about",
            "doesn't have.*information"
        ]
        
        if any(re.search(pattern, answer_lower) for pattern in simple_rejections):
            return True
    
    return False

def find_related_content(question: str, chunks: List[str]) -> List[str]:
    """Find content that might be related even if not directly matching"""
    question_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', question.lower()))
    question_keywords = extract_keywords(question)
    
    related_chunks = []
    
    for chunk in chunks:
        chunk_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', chunk.lower()))
        chunk_keywords = extract_keywords(chunk)
        
        # Calculate different types of similarity
        word_overlap = len(question_words & chunk_words)
        keyword_overlap = len(question_keywords & chunk_keywords)
        
        # Semantic similarity (look for related concepts)
        semantic_score = calculate_semantic_similarity(question, chunk)
        
        # Combined score
        total_score = word_overlap + (keyword_overlap * 2) + (semantic_score * 3)
        
        if total_score > 2:  # Lower threshold for related content
            related_chunks.append((chunk, total_score))
    
    # Sort by relevance and return top chunks
    related_chunks.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, score in related_chunks[:Config.MAX_RELEVANT_CHUNKS]]

def extract_keywords(text: str) -> set:
    """Extract important keywords from text"""
    # Common important words that should be weighted higher
    important_patterns = [
        r'\b(?:waiting|period|time|duration|days|months|years)\b',
        r'\b(?:disease|condition|illness|medical|health)\b',
        r'\b(?:pre-existing|existing|prior|previous)\b',
        r'\b(?:coverage|benefit|claim|policy|insurance)\b',
        r'\b(?:exclusion|limitation|restriction)\b',
        r'\b(?:premium|cost|fee|payment)\b',
        r'\b(?:eligibility|qualification|requirement)\b'
    ]
    
    keywords = set()
    for pattern in important_patterns:
        matches = re.findall(pattern, text.lower())
        keywords.update(matches)
    
    return keywords

def is_context_relevant_to_question(question: str, context: str) -> bool:
    """Check if the context is actually relevant to the question domain"""
    question_lower = question.lower()
    context_lower = context.lower()
    
    # Define question domains and their keywords
    question_domains = {
        'insurance': ['insurance', 'policy', 'coverage', 'claim', 'premium', 'deductible', 'waiting period', 'pre-existing', 'health insurance', 'life insurance'],
        'programming': ['python', 'javascript', 'programming', 'code', 'function', 'variable', 'algorithm', 'software'],
        'ai_ml': ['machine learning', 'artificial intelligence', 'neural network', 'deep learning', 'ai', 'ml', 'algorithm', 'model'],
        'technology': ['computer', 'software', 'hardware', 'technology', 'internet', 'digital'],
        'finance': ['bank', 'money', 'investment', 'financial', 'loan', 'credit', 'interest'],
        'medical': ['health', 'medical', 'disease', 'treatment', 'medicine', 'doctor', 'hospital']
    }
    
    # Identify question domain
    question_domain = None
    for domain, keywords in question_domains.items():
        if any(keyword in question_lower for keyword in keywords):
            question_domain = domain
            break
    
    # If we can't identify the question domain, assume context might be relevant
    if not question_domain:
        return True
    
    # Check if context has any keywords from the question domain
    domain_keywords = question_domains[question_domain]
    context_has_domain_keywords = any(keyword in context_lower for keyword in domain_keywords)
    
    # Special case: if context is about software development and question is about other programming concepts
    if question_domain == 'programming':
        software_dev_keywords = ['visual studio', 'vs code', 'editor', 'ide', 'development', 'coding', 'github']
        if any(keyword in context_lower for keyword in software_dev_keywords):
            # Check if question is about general programming concepts vs specific tools
            general_programming = ['what is python', 'what is javascript', 'how does', 'what are the benefits']
            if any(phrase in question_lower for phrase in general_programming):
                return False  # General programming questions not answerable from tool-specific context
    
    return context_has_domain_keywords


def calculate_semantic_similarity(text1: str, text2: str) -> float:
    """Simple semantic similarity based on common concepts"""
    # Insurance/medical concept groups
    concept_groups = [
        ['waiting', 'period', 'time', 'duration', 'wait'],
        ['disease', 'condition', 'illness', 'medical', 'health', 'sickness'],
        ['pre-existing', 'existing', 'prior', 'previous', 'before'],
        ['coverage', 'benefit', 'claim', 'policy', 'insurance', 'cover'],
        ['exclusion', 'limitation', 'restriction', 'exclude', 'limit'],
        ['premium', 'cost', 'fee', 'payment', 'price', 'amount']
    ]
    
    text1_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', text1.lower()))
    text2_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', text2.lower()))
    
    concept_matches = 0
    for group in concept_groups:
        group_set = set(group)
        if (text1_words & group_set) and (text2_words & group_set):
            concept_matches += 1
    
    return concept_matches / len(concept_groups)


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

@app.get("/test-document")
async def test_document_extraction():
    """Test document extraction with a sample document"""
    test_urls = [
        "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
        "https://raw.githubusercontent.com/microsoft/vscode/main/README.md"
    ]
    
    results = {}
    for url in test_urls:
        try:
            doc_text = await fetch_document(url)
            results[url] = {
                "success": True,
                "length": len(doc_text),
                "preview": doc_text[:500] + "..." if len(doc_text) > 500 else doc_text,
                "word_count": len(doc_text.split()),
                "meaningful_words": len(re.findall(r'\b[a-zA-Z]{3,}\b', doc_text))
            }
        except Exception as e:
            results[url] = {
                "success": False,
                "error": str(e)
            }
    
    return results

@app.post("/test-extraction")
async def test_custom_document(request: dict):
    """Test extraction with a custom document URL"""
    try:
        url = request.get("url", "")
        if not url:
            raise HTTPException(400, "URL is required")
        
        doc_text = await fetch_document(url)
        
        # Analyze the content
        analysis = {
            "url": url,
            "success": True,
            "total_length": len(doc_text),
            "word_count": len(doc_text.split()),
            "meaningful_words": len(re.findall(r'\b[a-zA-Z]{3,}\b', doc_text)),
            "has_pdf_metadata": "PDF" in doc_text[:200].upper(),
            "preview_start": doc_text[:300] + "..." if len(doc_text) > 300 else doc_text,
            "preview_end": "..." + doc_text[-300:] if len(doc_text) > 300 else "",
        }
        
        # Test chunking
        chunks = smart_chunk_text(doc_text)
        analysis["chunk_count"] = len(chunks)
        analysis["avg_chunk_size"] = sum(len(chunk) for chunk in chunks) / len(chunks) if chunks else 0
        
        return analysis
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "url": request.get("url", "")
        }

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
                
                # Get highly relevant chunks
                highly_relevant = [chunk for chunk, sim in similarities[:Config.MAX_RELEVANT_CHUNKS] 
                                 if sim > Config.SIMILARITY_THRESHOLD]
                
                # Get moderately relevant chunks if highly relevant ones are few
                if len(highly_relevant) < 2:
                    moderately_relevant = [chunk for chunk, sim in similarities[:Config.MAX_RELEVANT_CHUNKS * 2] 
                                         if sim > Config.SIMILARITY_THRESHOLD * 0.7]
                    highly_relevant.extend(moderately_relevant[:Config.MAX_RELEVANT_CHUNKS - len(highly_relevant)])
                
                # Find related content using keyword and semantic matching
                related_content = find_related_content(question, chunks)
                
                # Combine relevant chunks with related content
                all_relevant_chunks = highly_relevant.copy()
                for related_chunk in related_content:
                    if related_chunk not in all_relevant_chunks:
                        all_relevant_chunks.append(related_chunk)
                
                # Limit to max chunks
                all_relevant_chunks = all_relevant_chunks[:Config.MAX_RELEVANT_CHUNKS]
                
                # Check if the found context is actually relevant to the question domain
                if all_relevant_chunks:
                    combined_context = " ".join(all_relevant_chunks)
                    context_is_relevant = is_context_relevant_to_question(question, combined_context)
                    
                    if context_is_relevant:
                        # Generate answer using the relevant context
                        answer = await generate_answer(question, combined_context, has_relevant_context=True)
                        
                        # Check if the answer is generic/unhelpful
                        if is_generic_response(answer):
                            # Context wasn't actually helpful, try general knowledge
                            if Config.ENABLE_GENERAL_KNOWLEDGE:
                                general_answer = await call_general_knowledge_api(question)
                                if general_answer:
                                    answers.append(general_answer)
                                else:
                                    answers.append("I couldn't find relevant information in the provided document to answer this question.")
                            else:
                                answers.append("I couldn't find relevant information in the provided document to answer this question.")
                        else:
                            answers.append(answer)
                    else:
                        # Context is not relevant to question domain, use general knowledge directly
                        if Config.ENABLE_GENERAL_KNOWLEDGE:
                            general_answer = await call_general_knowledge_api(question)
                            if general_answer:
                                answers.append(general_answer)
                            else:
                                answers.append("I couldn't find relevant information in the provided document to answer this question.")
                        else:
                            answers.append("I couldn't find relevant information in the provided document to answer this question.")
                else:
                    # No relevant context found - try keyword-based search
                    question_words = set(re.findall(r'\w+', question.lower()))
                    best_chunk = ""
                    best_score = 0
                    
                    for chunk in chunks:
                        chunk_words = set(re.findall(r'\w+', chunk.lower()))
                        overlap = len(question_words & chunk_words)
                        if overlap > best_score:
                            best_score = overlap
                            best_chunk = chunk
                    
                    if best_chunk and best_score > 1:
                        # Found some keyword overlap
                        answer = await generate_answer(question, best_chunk, has_relevant_context=True)
                        answers.append(answer)
                    else:
                        # No relevant content in document - use general knowledge
                        if Config.ENABLE_GENERAL_KNOWLEDGE:
                            general_answer = await call_general_knowledge_api(question)
                            if general_answer:
                                answers.append(general_answer)
                            else:
                                answers.append("I couldn't find relevant information in the provided document to answer this question.")
                        else:
                            answers.append("I couldn't find relevant information in the provided document to answer this question.")
                        
            except Exception as e:
                print(f"Error processing question '{question}': {e}")
                # Try to get a general knowledge answer even if processing fails
                try:
                    if Config.ENABLE_GENERAL_KNOWLEDGE:
                        general_answer = await call_general_knowledge_api(question)
                        if general_answer:
                            answers.append(general_answer)
                        else:
                            answers.append("I encountered an error processing this question.")
                    else:
                        answers.append("I encountered an error processing this question.")
                except:
                    answers.append("I encountered an error processing this question.")
        
        return RunResponse(answers=answers)
        
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Processing failed: {e}")
        raise HTTPException(500, f"An internal error occurred: {e}")

# For Vercel
app = app
