"""
Multi-LLM API with fallback providers: OpenAI, Anthropic Claude, and Hugging Face
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

def chunk_text(text: str) -> List[Dict[str, Any]]:
    """Smart text chunking for any document type."""
    chunk_size = 800  # Optimal size for most document types
    overlap = 150
    chunks = []
    
    # Clean the text first
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = text.strip()
    
    if len(text) < chunk_size:
        # If document is small, return as single chunk
        return [{
            'text': text,
            'score': 0.5,
            'metadata': {'chunk_id': 0, 'type': 'full_document'}
        }]
    
    # Strategy 1: Try section-based chunking (headers, titles, etc.)
    section_patterns = [
        r'\n\s*(?:Chapter|Section|Part|Article)\s+\d+.*?\n',
        r'\n\s*\d+\.\s+[A-Z][^.\n]*\n',
        r'\n\s*[A-Z][A-Z\s]{10,}\n',  # All caps headers
        r'\n\s*\*{2,}.*?\*{2,}\s*\n',  # Headers with asterisks
        r'\n\s*={3,}\s*\n',  # Lines with equals signs
        r'\n\s*-{3,}\s*\n'   # Lines with dashes
    ]
    
    sections = []
    for pattern in section_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            sections = re.split(pattern, text, flags=re.IGNORECASE)
            break
    
    if len(sections) > 3:  # Good section structure found
        current_chunk = ""
        for section in sections:
            section = section.strip()
            if not section:
                continue
                
            # If adding this section would exceed chunk size, save current chunk
            if len(current_chunk) + len(section) > chunk_size and current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'score': 0.6,
                    'metadata': {'chunk_id': len(chunks), 'type': 'section_based'}
                })
                # Start new chunk with overlap
                current_chunk = current_chunk[-overlap:] + " " + section
            else:
                current_chunk += " " + section if current_chunk else section
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'score': 0.6,
                'metadata': {'chunk_id': len(chunks), 'type': 'section_based'}
            })
    
    # Strategy 2: Paragraph-based chunking
    if len(chunks) < 3:
        chunks = []
        paragraphs = re.split(r'\n\s*\n', text)
        
        if len(paragraphs) > 5:  # Good paragraph structure
            current_chunk = ""
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                    
                # If adding this paragraph would exceed chunk size, save current chunk
                if len(current_chunk) + len(para) > chunk_size and current_chunk:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'score': 0.5,
                        'metadata': {'chunk_id': len(chunks), 'type': 'paragraph_based'}
                    })
                    # Start new chunk with overlap
                    current_chunk = current_chunk[-overlap:] + " " + para
                else:
                    current_chunk += " " + para if current_chunk else para
            
            # Add the last chunk
            if current_chunk.strip():
                chunks.append({
                    'text': current_chunk.strip(),
                    'score': 0.5,
                    'metadata': {'chunk_id': len(chunks), 'type': 'paragraph_based'}
                })
    
    # Strategy 3: Sentence-based chunking (for documents with good sentence structure)
    if len(chunks) < 3:
        chunks = []
        sentences = re.split(r'[.!?]+', text)
        
        current_chunk = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue
                
            # If adding this sentence would exceed chunk size, save current chunk
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append({
                    'text': current_chunk.strip() + '.',
                    'score': 0.4,
                    'metadata': {'chunk_id': len(chunks), 'type': 'sentence_based'}
                })
                # Start new chunk with overlap
                overlap_sentences = current_chunk.split('.')[-2:]  # Last 2 sentences for overlap
                current_chunk = '. '.join(overlap_sentences) + ". " + sentence
            else:
                current_chunk += ". " + sentence if current_chunk else sentence
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip() + '.',
                'score': 0.4,
                'metadata': {'chunk_id': len(chunks), 'type': 'sentence_based'}
            })
    
    # Strategy 4: Fixed-size word chunking (fallback)
    if len(chunks) < 2:
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append({
                'text': chunk_text,
                'score': 0.3,
                'metadata': {'chunk_id': len(chunks), 'type': 'word_based'}
            })
    
    # Ensure we have at least one chunk
    if not chunks:
        chunks = [{
            'text': text[:2000],  # Take first 2000 chars as fallback
            'score': 0.2,
            'metadata': {'chunk_id': 0, 'type': 'fallback'}
        }]
    
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
            chunk_copy = chunk.copy()
            chunk_copy['score'] = final_score
            scored_chunks.append(chunk_copy)
    
    # Sort by score and return top 5 for better context
    scored_chunks.sort(key=lambda x: x['score'], reverse=True)
    return scored_chunks[:5]

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
    """Generate a fallback answer using intelligent text processing for any document type."""
    if not relevant_chunks:
        return "Information not available in the provided document."
    
    # Combine all relevant chunks for better context
    context = " ".join([chunk['text'] for chunk in relevant_chunks])
    question_lower = question.lower()
    
    # Generic pattern matching for common question types
    
    # Time period questions (days, months, years)
    time_patterns = [
        (r'(\d+)\s*days?', 'days'),
        (r'(\d+)\s*months?', 'months'), 
        (r'(\d+)\s*years?', 'years'),
        (r'(\d+)\s*weeks?', 'weeks')
    ]
    
    # Look for numerical values and percentages
    if any(word in question_lower for word in ['how much', 'percentage', 'rate', 'amount', 'cost', 'fee']):
        # Extract percentages
        percentage_matches = re.findall(r'(\d+(?:\.\d+)?)%', context)
        if percentage_matches:
            # Find the most relevant percentage based on context
            for percentage in percentage_matches:
                context_around = ""
                for sentence in re.split(r'[.!?]+', context):
                    if f"{percentage}%" in sentence:
                        context_around = sentence.strip()
                        break
                if context_around:
                    return f"The rate/percentage is {percentage}%. {context_around[:200]}..."
        
        # Extract monetary amounts
        money_matches = re.findall(r'(?:INR|USD|\$|₹)\s*[\d,]+(?:\.\d+)?', context, re.IGNORECASE)
        if money_matches:
            return f"The amount mentioned is {money_matches[0]}. Additional details may be available in the document."
    
    # Time-related questions
    if any(word in question_lower for word in ['when', 'period', 'duration', 'time', 'deadline']):
        for pattern, unit in time_patterns:
            matches = re.findall(pattern, context, re.IGNORECASE)
            if matches:
                # Find context around the time period
                for match in matches:
                    context_sentences = re.split(r'[.!?]+', context)
                    for sentence in context_sentences:
                        if f"{match} {unit}" in sentence.lower() or f"{match}{unit}" in sentence.lower():
                            return f"The time period is {match} {unit}. {sentence.strip()}"
    
    # Yes/No questions
    if question_lower.startswith(('is ', 'are ', 'does ', 'do ', 'can ', 'will ', 'should ')):
        # Look for affirmative/negative indicators
        positive_indicators = ['yes', 'covered', 'included', 'available', 'provided', 'allowed', 'eligible']
        negative_indicators = ['no', 'not covered', 'excluded', 'not available', 'not provided', 'not allowed']
        
        context_lower = context.lower()
        positive_count = sum(1 for indicator in positive_indicators if indicator in context_lower)
        negative_count = sum(1 for indicator in negative_indicators if indicator in context_lower)
        
        if positive_count > negative_count:
            # Find the most relevant positive sentence
            for sentence in re.split(r'[.!?]+', context):
                if any(indicator in sentence.lower() for indicator in positive_indicators):
                    return f"Yes, {sentence.strip().lower()}"
        elif negative_count > positive_count:
            # Find the most relevant negative sentence
            for sentence in re.split(r'[.!?]+', context):
                if any(indicator in sentence.lower() for indicator in negative_indicators):
                    return f"No, {sentence.strip().lower()}"
    
    # Definition questions
    if any(word in question_lower for word in ['what is', 'define', 'definition', 'meaning', 'means']):
        # Look for definition patterns
        definition_patterns = [
            r'means\s+(.{20,200}?)[\.\n]',
            r'is\s+defined\s+as\s+(.{20,200}?)[\.\n]',
            r'refers\s+to\s+(.{20,200}?)[\.\n]',
            r':\s*(.{20,200}?)[\.\n]'
        ]
        
        for pattern in definition_patterns:
            matches = re.findall(pattern, context, re.IGNORECASE | re.DOTALL)
            if matches:
                definition = matches[0].strip()
                return f"{definition}."
    
    # Generic smart sentence extraction
    question_keywords = set(re.findall(r'\w+', question_lower))
    # Remove common stop words
    stop_words = {'what', 'is', 'the', 'are', 'does', 'do', 'can', 'will', 'how', 'when', 'where', 'why', 'which'}
    question_keywords = question_keywords - stop_words
    
    sentences = re.split(r'[.!?]+', context)
    scored_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 15 or len(sentence) > 400:
            continue
            
        # Skip sentences that look like table headers or metadata
        if (sentence.count('|') > 3 or 
            sentence.count('INR') > 2 or 
            sentence.count('Up to SI') > 1 or
            sentence.count('Page') > 0 and sentence.count('of') > 0):
            continue
            
        sentence_words = set(re.findall(r'\w+', sentence.lower()))
        
        # Calculate relevance score
        keyword_overlap = len(question_keywords.intersection(sentence_words))
        
        # Boost score for sentences with key indicators
        content_boost = 0
        if any(word in sentence.lower() for word in ['including', 'such as', 'provided that', 'subject to']):
            content_boost += 0.5
        if any(word in sentence.lower() for word in ['shall', 'must', 'required', 'mandatory']):
            content_boost += 0.3
        if re.search(r'\d+', sentence):  # Contains numbers
            content_boost += 0.2
            
        final_score = keyword_overlap + content_boost
        
        if final_score > 0:
            scored_sentences.append((sentence, final_score))
    
    # Sort by score and return the best match
    if scored_sentences:
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        best_sentence = scored_sentences[0][0]
        
        # Clean up the sentence
        if not best_sentence.endswith('.'):
            best_sentence += '.'
            
        return best_sentence
    
    # Final fallback - return a summary of the most relevant chunk
    if relevant_chunks:
        best_chunk = relevant_chunks[0]['text']
        # Try to find a complete sentence from the chunk
        sentences = re.split(r'[.!?]+', best_chunk)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 30 and len(sentence) < 300:
                if not sentence.endswith('.'):
                    sentence += '.'
                return sentence
    
    return "The specific information requested is not clearly available in the provided document."

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
        
        # Chunk text
        chunks = chunk_text(document_text)
        
        # Process questions
        answers = []
        
        for question in request_body.questions:
            # Search for relevant chunks
            relevant_chunks = simple_search(question, chunks)
            
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
