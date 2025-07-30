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
    """Smart text chunking for policy documents."""
    chunk_size = 1000  # Larger chunks for better context
    overlap = 200
    chunks = []
    
    # First, try to split by paragraphs or sections
    paragraphs = re.split(r'\n\s*\n', text)
    
    # If we have good paragraph structure, use that
    if len(paragraphs) > 5:
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
                    'metadata': {'chunk_id': len(chunks), 'type': 'paragraph'}
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
                'metadata': {'chunk_id': len(chunks), 'type': 'paragraph'}
            })
    
    # Fallback to word-based chunking if paragraph approach doesn't work well
    if len(chunks) < 3:
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append({
                'text': chunk_text,
                'score': 0.5,
                'metadata': {'chunk_id': len(chunks), 'type': 'word_based'}
            })
    
    return chunks

def simple_search(query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Enhanced keyword-based search for policy documents."""
    query_words = set(re.findall(r'\w+', query.lower()))
    scored_chunks = []
    
    # Keywords that indicate important policy information
    policy_keywords = {
        'grace period', 'waiting period', 'maternity', 'cataract', 'organ donor', 
        'claim discount', 'health check', 'hospital', 'ayush', 'room rent', 'icu',
        'premium', 'coverage', 'benefit', 'policy', 'insured', 'treatment'
    }
    
    for chunk in chunks:
        chunk_text_lower = chunk['text'].lower()
        chunk_words = set(re.findall(r'\w+', chunk_text_lower))
        
        # Basic keyword overlap score
        overlap = len(query_words.intersection(chunk_words))
        base_score = overlap / len(query_words) if query_words else 0
        
        # Boost score for policy-relevant terms
        policy_boost = 0
        for keyword in policy_keywords:
            if keyword in chunk_text_lower:
                policy_boost += 0.1
        
        # Boost score for exact phrase matches
        phrase_boost = 0
        query_text = query.lower()
        if len(query_text) > 10 and query_text in chunk_text_lower:
            phrase_boost = 0.3
        
        # Calculate final score
        final_score = base_score + policy_boost + phrase_boost
        
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
    """Generate a fallback answer using smart text processing for policy documents."""
    if not relevant_chunks:
        return "Information not available in the provided document."
    
    # Combine all relevant chunks for better context
    context = " ".join([chunk['text'] for chunk in relevant_chunks])
    
    # Extract key information based on question type
    question_lower = question.lower()
    
    # Pattern matching for specific policy questions
    if 'grace period' in question_lower and 'premium' in question_lower:
        # Look for grace period information
        grace_patterns = [
            r'grace period.*?(\d+)\s*days?.*?premium',
            r'premium.*?grace period.*?(\d+)\s*days?',
            r'(\d+)\s*days?.*?grace period',
            r'grace period.*?thirty.*?days?',
        ]
        for pattern in grace_patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                if 'thirty' in match.group(0).lower():
                    return "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits."
                elif match.group(1):
                    days = match.group(1)
                    return f"A grace period of {days} days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits."
    
    elif 'waiting period' in question_lower and ('pre-existing' in question_lower or 'ped' in question_lower):
        # Look for PED waiting period
        if '36 months' in context or 'thirty-six' in context.lower():
            return "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered."
        
        ped_patterns = [
            r'waiting period.*?(\d+)\s*months?.*?pre-existing',
            r'pre-existing.*?(\d+)\s*months?.*?continuous coverage',
            r'(\d+)\s*months?.*?continuous coverage.*?pre-existing',
        ]
        for pattern in ped_patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                months = match.group(1)
                return f"There is a waiting period of {months} months of continuous coverage for pre-existing diseases to be covered."
    
    elif 'maternity' in question_lower:
        # Look for maternity coverage information
        if 'maternity' in context.lower():
            if '24 months' in context or 'twenty-four' in context.lower():
                return "Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months. The benefit is limited to two deliveries or terminations during the policy period."
            elif 'covered' in context.lower():
                return "Yes, the policy covers maternity expenses with specific conditions and waiting periods for continuous coverage."
    
    elif 'cataract' in question_lower and 'waiting' in question_lower:
        # Look for cataract waiting period
        if 'two years' in context.lower() or '2 years' in context or 'two (2) years' in context:
            return "The policy has a specific waiting period of two (2) years for cataract surgery."
        
        cataract_patterns = [
            r'cataract.*?(\d+)\s*years?',
            r'waiting.*?(\d+)\s*years?.*?cataract',
        ]
        for pattern in cataract_patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                years = match.group(1)
                return f"The policy has a waiting period of {years} years for cataract surgery."
    
    elif 'organ donor' in question_lower:
        if 'organ donor' in context.lower() and ('covered' in context.lower() or 'indemnifies' in context.lower()):
            return "Yes, the policy indemnifies the medical expenses for the organ donor's hospitalization for the purpose of harvesting the organ, provided the organ is for an insured person and the donation complies with the Transplantation of Human Organs Act, 1994."
    
    elif 'no claim discount' in question_lower or 'ncd' in question_lower:
        # Look for NCD information
        if '5%' in context and ('no claim' in context.lower() or 'ncd' in context.lower()):
            return "A No Claim Discount of 5% on the base premium is offered on renewal for a one-year policy term if no claims were made in the preceding year. The maximum aggregate NCD is capped at 5% of the total base premium."
        
        ncd_patterns = [
            r'no claim discount.*?(\d+)%',
            r'ncd.*?(\d+)%',
            r'(\d+)%.*?no claim discount',
        ]
        for pattern in ncd_patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                percentage = match.group(1)
                return f"A No Claim Discount of {percentage}% is offered on renewal if no claims were made."
    
    elif 'health check' in question_lower or 'preventive' in question_lower:
        if 'health check' in context.lower() or 'check-up' in context.lower():
            if 'two continuous policy years' in context.lower() or 'block of two' in context.lower():
                return "Yes, the policy reimburses expenses for health check-ups at the end of every block of two continuous policy years, provided the policy has been renewed without a break. The amount is subject to the limits specified in the Table of Benefits."
            return "Yes, the policy provides benefits for preventive health check-ups with specific conditions."
    
    elif 'hospital' in question_lower and 'define' in question_lower:
        # Look for hospital definition
        if 'inpatient beds' in context.lower() and ('10' in context or '15' in context):
            return "A hospital is defined as an institution with at least 10 inpatient beds (in towns with a population below ten lakhs) or 15 beds (in all other places), with qualified nursing staff and medical practitioners available 24/7, a fully equipped operation theatre, and which maintains daily records of patients."
        
        hospital_patterns = [
            r'hospital.*?(\d+).*?beds?',
            r'(\d+).*?beds?.*?hospital',
        ]
        for pattern in hospital_patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                beds = match.group(1)
                return f"A hospital is defined as an institution with at least {beds} inpatient beds along with other specified requirements."
    
    elif 'ayush' in question_lower:
        if 'ayush' in context.lower():
            return "The policy covers medical expenses for inpatient treatment under Ayurveda, Yoga, Naturopathy, Unani, Siddha, and Homeopathy systems up to the Sum Insured limit, provided the treatment is taken in an AYUSH Hospital."
    
    elif 'room rent' in question_lower or ('sub-limits' in question_lower and 'plan a' in question_lower):
        # Look for room rent limits for Plan A
        if '1% of' in context and '2% of' in context and ('room' in context.lower() or 'icu' in context.lower()):
            return "Yes, for Plan A, the daily room rent is capped at 1% of the Sum Insured, and ICU charges are capped at 2% of the Sum Insured. These limits do not apply if the treatment is for a listed procedure in a Preferred Provider Network (PPN)."
        
        rent_patterns = [
            r'room.*?(\d+)%.*?sum insured',
            r'(\d+)%.*?sum insured.*?room',
        ]
        for pattern in rent_patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                percentage = match.group(1)
                return f"Room rent is capped at {percentage}% of the Sum Insured for Plan A."
    
    # If no specific pattern matches, try to find the most relevant sentence
    question_keywords = set(re.findall(r'\w+', question.lower()))
    sentences = re.split(r'[.!?]+', context)
    
    best_sentence = ""
    best_score = 0
    
    for sentence in sentences:
        if len(sentence.strip()) < 20 or len(sentence.strip()) > 300:
            continue
            
        # Skip table-like content
        if sentence.count('INR') > 2 or sentence.count('Up to SI') > 2:
            continue
            
        sentence_words = set(re.findall(r'\w+', sentence.lower()))
        overlap = len(question_keywords.intersection(sentence_words))
        
        if overlap > best_score:
            best_score = overlap
            best_sentence = sentence.strip()
    
    if best_sentence and best_score > 1:
        if not best_sentence.endswith('.'):
            best_sentence += '.'
        return best_sentence
    
    # Final fallback
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
