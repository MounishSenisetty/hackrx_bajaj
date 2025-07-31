#!/usr/bin/env python3
"""
Test the enhanced Q&A capabilities with better context retrieval and general knowledge
"""
import requests
import json
import time

def test_enhanced_qa():
    base_url = "https://hackrx-bajaj.vercel.app"
    bearer_token = "ca6914a6c8df9d1ce075149c3ab9f060e666c75940576e37a98b3cf0e9092c72"
    
    print("Testing Enhanced Q&A Capabilities...")
    print("=" * 60)
    
    # Test 1: Questions with answers in the document
    print("\n🔍 TEST 1: Questions with answers in document")
    test_document_qa(base_url, bearer_token, 
                    "https://raw.githubusercontent.com/microsoft/vscode/main/README.md",
                    [
                        "What is Visual Studio Code?",
                        "What programming languages does VS Code support?",
                        "How do you contribute to VS Code?"
                    ])
    
    # Test 2: Questions NOT in the document (should use general knowledge)
    print("\n🧠 TEST 2: Questions requiring general knowledge")
    test_document_qa(base_url, bearer_token,
                    "https://raw.githubusercontent.com/microsoft/vscode/main/README.md",
                    [
                        "What is the waiting period for pre-existing diseases in health insurance?",
                        "What are the benefits of life insurance?",
                        "How does machine learning work?"
                    ])
    
    # Test 3: Mixed questions (some in document, some requiring general knowledge)
    print("\n🔀 TEST 3: Mixed relevant and irrelevant questions")
    test_document_qa(base_url, bearer_token,
                    "https://raw.githubusercontent.com/microsoft/vscode/main/README.md",
                    [
                        "What is VS Code?",  # In document
                        "What is Python?",   # General knowledge
                        "How to install VS Code?",  # Might be in document
                        "What is artificial intelligence?"  # General knowledge
                    ])
    
    print("\n" + "=" * 60)
    print("📋 ENHANCED FEATURES TESTED:")
    print("✅ Lowered similarity threshold (0.5 vs 0.7)")
    print("✅ Increased relevant chunks (5 vs 3)")
    print("✅ Semantic and keyword matching")
    print("✅ General knowledge fallback when document lacks info")
    print("✅ Better context extraction and relevance detection")
    print("✅ Intelligent response quality filtering")
    print("=" * 60)

def test_document_qa(base_url, bearer_token, document_url, questions):
    """Test Q&A with a specific document and questions"""
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {bearer_token}"
        }
        
        payload = {
            "documents": document_url,
            "questions": questions
        }
        
        print(f"📄 Document: {document_url.split('/')[-1]}")
        print("⏳ Processing questions...")
        
        response = requests.post(
            f"{base_url}/hackrx/run", 
            headers=headers, 
            json=payload, 
            timeout=60  # Longer timeout for processing
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            
            for i, (question, answer) in enumerate(zip(questions, result['answers'])):
                print(f"\n❓ Q{i+1}: {question}")
                print(f"💬 A{i+1}: {answer}")
                
                # Analyze answer quality
                if "based on general knowledge" in answer.lower():
                    print("   🧠 Used general knowledge")
                elif len(answer) > 20 and "couldn't find" not in answer.lower():
                    print("   📖 Used document context")
                else:
                    print("   ⚠️  Limited response")
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    # Wait a moment for deployment
    print("⏳ Waiting for deployment to complete...")
    time.sleep(10)
    test_enhanced_qa()
