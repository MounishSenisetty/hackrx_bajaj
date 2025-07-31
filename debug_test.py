#!/usr/bin/env python3
"""
Debug the policy document processing step by step
"""
import requests
import json

def test_single_question():
    base_url = "https://hackrx-bajaj.vercel.app"
    bearer_token = "ca6914a6c8df9d1ce075149c3ab9f060e666c75940576e37a98b3cf0e9092c72"
    
    print("Debug Test: Single Question Processing")
    print("=" * 50)
    
    # Test with a simple document first (text file)
    simple_doc = "https://raw.githubusercontent.com/microsoft/vscode/main/README.md"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {bearer_token}"
    }
    
    # Test 1: Simple question with simple document
    print("\n🔍 TEST 1: Simple document + simple question")
    payload = {
        "documents": simple_doc,
        "questions": ["What is Visual Studio Code?"]
    }
    
    try:
        response = requests.post(f"{base_url}/hackrx/run", headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success: {result['answers'][0]}")
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    # Test 2: Policy PDF with one simple question
    print("\n🔍 TEST 2: Policy PDF + one question")
    policy_doc = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    
    payload = {
        "documents": policy_doc,
        "questions": ["What is the waiting period for pre-existing diseases?"]
    }
    
    try:
        response = requests.post(f"{base_url}/hackrx/run", headers=headers, json=payload, timeout=90)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success: {result['answers'][0]}")
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    test_single_question()
