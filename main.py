import json
import httpx # Added for making async HTTP requests
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import asyncio # Added for concurrent processing

# Initialize FastAPI app
app = FastAPI(
    title="LLM-Powered Intelligent Query-Retrieval System",
    description="API for processing natural language queries against documents and making contextual decisions."
)

# --- Document Content Fetching (Simulated for complex formats) ---
async def fetch_document_content(url: str) -> str:
    """
    Fetches content from a given URL.
    NOTE: For actual PDF/DOCX/email parsing, you would integrate libraries
    like PyPDF2, python-docx, or email parsers here.
    For this submission, it attempts to fetch raw text content.
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client: # Increased timeout for external fetches
            response = await client.get(url)
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
            # Attempt to decode as text. For binary formats like PDF, this will be raw bytes.
            # A real system would then pass these bytes to a PDF parser.
            return response.text
    except httpx.RequestError as exc:
        print(f"An error occurred while requesting {exc.request.url!r}: {exc}")
        return f"Error fetching document from URL: {url}. Details: {exc}"
    except Exception as e:
        print(f"An unexpected error occurred during document fetch: {e}")
        return f"Unexpected error fetching document: {e}"

# --- Request and Response Models for FastAPI ---
class RunRequest(BaseModel):
    documents: str = Field(..., description="URL to the PDF Blob or other document type.")
    questions: List[str] = Field(..., description="List of natural language questions to ask about the documents.")

class RunResponse(BaseModel):
    answers: List[str] = Field(..., description="List of answers corresponding to the input questions.")

# --- LLM Processing Function ---
async def call_llm_for_answer(document_content: str, question: str) -> str:
    """
    Calls the LLM to get a concise answer for a given question based on document content.
    In a real system, this would involve semantic search (Pinecone/FAISS) to retrieve
    relevant chunks before sending to the LLM to optimize token usage.
    """
    prompt = f"""
      You are an intelligent document analysis assistant. Your task is to answer the following question concisely and directly, based on the provided policy document. If the document contains the answer, state it clearly. If the document does not contain the answer, state 'Information not available in the provided document.' Do not add any conversational filler, just the direct answer.

      Policy Document:
      {document_content}

      Question:
      "{question}"

      Concise Answer:
    """

    chat_history = []
    chat_history.append({"role": "user", "parts": [{"text": prompt}]})

    payload = {
        "contents": chat_history,
        "generationConfig": {
            "responseMimeType": "text/plain"
        }
    }

    # API Key for Gemini 2.0 Flash (managed by the environment)
    api_key = "" # Canvas will provide this at runtime
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

    try:
        async with httpx.AsyncClient(timeout=60.0) as client: # Increased timeout for LLM calls
            api_response = await client.post(api_url, json=payload, headers={'Content-Type': 'application/json'})
            api_response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
            result = api_response.json()

        if result.get("candidates") and result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts"):
            return result["candidates"][0]["content"]["parts"][0]["text"]
        else:
            print(f"LLM response structure unexpected: {result}")
            return "Information not available from LLM (unexpected response)."
    except httpx.RequestError as exc:
        print(f"An error occurred while calling LLM API {exc.request.url!r}: {exc}")
        return f"Error calling LLM API: {exc}"
    except Exception as e:
        print(f"An unexpected error occurred during LLM call: {e}")
        return f"Unexpected error during LLM call: {e}"


# --- API Endpoint ---
@app.post("/hackrx/run", response_model=RunResponse)
async def run_submissions(request_body: RunRequest, request: Request):
    """
    Processes a list of natural language questions against a document
    (simulated from a PDF Blob URL) and returns structured answers.
    """
    # --- Authentication ---
    auth_header = request.headers.get("Authorization")
    expected_token = "ca6914a6c8df9d1ce075149c3ab9f060e666c75940576e37a98b3cf0e9092c72"
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication required: Bearer token missing.")
    
    provided_token = auth_header.split(" ")[1]
    if provided_token != expected_token:
        raise HTTPException(status_code=403, detail="Invalid Bearer token.")

    # --- Document Processing (Fetch content from URL) ---
    # In a real system, this would involve robust PDF/DOCX/email parsing.
    # For this submission, we fetch the raw content and pass it to the LLM.
    document_content = await fetch_document_content(request_body.documents)
    
    if document_content.startswith("Error fetching document"):
        raise HTTPException(status_code=500, detail=document_content)

    # --- Process each question concurrently ---
    # Create a list of coroutines (tasks) for each question
    tasks = [call_llm_for_answer(document_content, question) for question in request_body.questions]
    
    # Run all tasks concurrently and collect results
    answers = await asyncio.gather(*tasks)

    # --- JSON Output ---
    return RunResponse(answers=answers)
