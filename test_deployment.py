#!/usr/bin/env python3
"""
Test script for the deployed API with improved PDF extraction
"""
import requests
import json
import time

def test_api():
    base_url = "https://hackrx-bajaj.vercel.app"
    bearer_token = "ca6914a6c8df9d1ce075149c3ab9f060e666c75940576e37a98b3cf0e9092c72"
    
    print("Testing deployed API with improved PDF extraction...")
    
    # Test health endpoint
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Response: {result}")
        
        if not result.get("bearer_token_set"):
            print("⚠️  WARNING: Bearer token not set on Vercel!")
        else:
            print("✅ Bearer token is configured")
            
    except Exception as e:
        print(f"❌ Health test failed: {e}")
        return
    
    # Test document extraction capabilities
    print("\n2. Testing document extraction...")
    try:
        response = requests.get(f"{base_url}/test-document", timeout=15)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            for url, analysis in result.items():
                print(f"\n📄 Document: {url}")
                if analysis.get("success"):
                    print(f"   ✅ Length: {analysis['length']} chars")
                    print(f"   ✅ Words: {analysis['word_count']}")
                    print(f"   ✅ Meaningful words: {analysis['meaningful_words']}")
                    print(f"   ✅ Preview: {analysis['preview'][:100]}...")
                else:
                    print(f"   ❌ Error: {analysis.get('error')}")
        else:
            print(f"❌ Error: {response.text}")
    except Exception as e:
        print(f"❌ Document test failed: {e}")
    
    # Test custom PDF extraction
    print("\n3. Testing custom PDF extraction...")
    test_pdfs = [
        "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
        "https://www.adobe.com/support/documentation/en/acrobat/acrobat_dc_sdk.pdf"
    ]
    
    for pdf_url in test_pdfs:
        try:
            print(f"\n📑 Testing PDF: {pdf_url}")
            response = requests.post(
                f"{base_url}/test-extraction",
                json={"url": pdf_url},
                timeout=20
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    print(f"   ✅ Extracted {result['total_length']} characters")
                    print(f"   ✅ {result['word_count']} words, {result['meaningful_words']} meaningful")
                    print(f"   ✅ {result['chunk_count']} chunks, avg size: {result['avg_chunk_size']:.0f}")
                    if result.get("has_pdf_metadata"):
                        print("   ⚠️  Contains PDF metadata (may need better extraction)")
                    print(f"   📖 Preview: {result['preview_start'][:150]}...")
                else:
                    print(f"   ❌ Failed: {result.get('error')}")
            else:
                print(f"   ❌ HTTP {response.status_code}: {response.text}")
        except Exception as e:
            print(f"   ❌ Exception: {e}")
    
    # Test main Q&A endpoint with PDF
    if result.get("bearer_token_set"):
        print("\n4. Testing main Q&A endpoint with PDF...")
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {bearer_token}"
            }
            
            payload = {
                "documents": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
                "questions": [
                    "What type of file is this?",
                    "What does this document contain?"
                ]
            }
            
            print("Sending Q&A request...")
            response = requests.post(
                f"{base_url}/hackrx/run", 
                headers=headers, 
                json=payload, 
                timeout=45  # Longer timeout for full processing
            )
            
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print("✅ Q&A successful!")
                for i, answer in enumerate(result['answers']):
                    print(f"   Q{i+1}: {payload['questions'][i]}")
                    print(f"   A{i+1}: {answer}")
            else:
                print(f"❌ Error: {response.text}")
                
        except Exception as e:
            print(f"❌ Main endpoint test failed: {e}")
    else:
        print("\n4. ⏭️  Skipping Q&A test - Bearer token not configured on Vercel")
    
    print("\n" + "="*60)
    print("📋 SUMMARY:")
    print("- Enhanced PDF extraction with PyMuPDF fallback")
    print("- Multiple text extraction methods")
    print("- Improved chunking and vector search")
    print("- Better error handling and debugging")
    print("="*60)

if __name__ == "__main__":
    test_api()
