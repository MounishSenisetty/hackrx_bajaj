#!/usr/bin/env python3

import asyncio
import httpx
import sys
sys.path.append('.')

from api.index import fetch_document, extract_pdf_text

async def test_pdf_processing():
    """Test PDF processing locally"""
    pdf_url = "https://i.postimg.cc/JzLNVgtB/NPMPP.pdf"
    
    print("Fetching document...")
    try:
        content = await fetch_document(pdf_url)
        print(f"Content type: {type(content)}")
        print(f"Content length: {len(content)}")
        print("First 500 characters:")
        print(repr(content[:500]))
        
        if content.startswith("%PDF"):
            print("\nDetected PDF format, trying to extract text...")
            try:
                import PyPDF2
                from io import BytesIO
                
                pdf_reader = PyPDF2.PdfReader(BytesIO(content))
                print(f"Number of pages: {len(pdf_reader.pages)}")
                
                text = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    print(f"Page {page_num + 1} text length: {len(page_text)}")
                    text += page_text + "\n"
                
                print(f"\nTotal extracted text length: {len(text)}")
                print("First 500 characters of extracted text:")
                print(repr(text[:500]))
                
            except Exception as e:
                print(f"Error extracting PDF text: {e}")
        else:
            print("Not a PDF file")
            
    except Exception as e:
        print(f"Error fetching document: {e}")

if __name__ == "__main__":
    asyncio.run(test_pdf_processing())
