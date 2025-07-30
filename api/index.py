
"""
Vercel-Compatible Document Query System (Lightweight, No Local ML)
"""


import os
import re
import io
import httpx
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


app = FastAPI(title="Vercel-Compatible Document Query System", version="1.0.0")

# Configuration
class Config:
    BEARER_TOKEN = os.getenv("BEARER_TOKEN", "ca6914a6c8df9d1ce075149c3ab9f060e666c75940576e37a98b3cf0e9092c72")
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
    HUGGINGFACE_MODEL = "google/flan-t5-base"
    REQUEST_TIMEOUT = 30.0
    MAX_DOCUMENT_SIZE = 10 * 1024 * 1024





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



# No local embeddings or vector search in Vercel-compatible version



# Vercel-compatible: Use HuggingFace Inference API for answers
async def call_huggingface_api(question: str, context: str = "") -> str:
    if not Config.HUGGINGFACE_API_KEY:
        return "HuggingFace API key not set."
    prompt = f"Answer the following question as accurately as possible.\n\nQuestion: {question}\nContext: {context}\nAnswer:"
    url = f"https://api-inference.huggingface.co/models/{Config.HUGGINGFACE_MODEL}"
    headers = {
        "Authorization": f"Bearer {Config.HUGGINGFACE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 200}}
    try:
        async with httpx.AsyncClient(timeout=Config.REQUEST_TIMEOUT) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
                return result[0]["generated_text"].replace(prompt, "").strip()
            elif isinstance(result, dict) and "generated_text" in result:
                return result["generated_text"].replace(prompt, "").strip()
            else:
                return str(result)
    except Exception as e:
        return f"Error calling HuggingFace API: {e}"


# API Endpoints
@app.get("/")
async def root():
    return {"message": "Vercel-Compatible Document Query System is running"}

@app.post("/hackrx/run", response_model=RunResponse)
async def run_submissions(req: RunRequest, http_req: Request):
    if http_req.headers.get("Authorization") != f"Bearer {Config.BEARER_TOKEN}":
        raise HTTPException(403, "Invalid Bearer token")
    try:
        doc_text = await fetch_document(req.documents)
        if not doc_text or len(doc_text.strip()) < 10:
            raise HTTPException(400, "Failed to extract content from document")
        # For Vercel: just use the full doc as context for each question
        answers = [await call_huggingface_api(q, doc_text) for q in req.questions]
        return RunResponse(answers=answers)
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Processing failed: {e}")
        raise HTTPException(500, f"An internal error occurred: {e}")

# For Vercel
app = app
