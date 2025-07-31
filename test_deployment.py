#!/usr/bin/env python3
"""
Test script for the deployed API
"""
import requests
import json
import time

def test_api():
    base_url = "https://hackrx-bajaj.vercel.app"
    bearer_token = "ca6914a6c8df9d1ce075149c3ab9f060e666c75940576e37a98b3cf0e9092c72"
    
    print("Testing deployed API...")
    
    # Test health endpoint
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Health test failed: {e}")
    
    # Test API connectivity
    print("\n2. Testing API connectivity...")
    try:
        response = requests.get(f"{base_url}/test-apis", timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"API test failed: {e}")
    
    # Test main endpoint with a simple document
    print("\n3. Testing main Q&A endpoint...")
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {bearer_token}"
        }
        
        payload = {
            "documents": "https://raw.githubusercontent.com/microsoft/vscode/main/README.md",
            "questions": ["What is Visual Studio Code?"]
        }
        
        response = requests.post(
            f"{base_url}/hackrx/run", 
            headers=headers, 
            json=payload, 
            timeout=30
        )
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Answer: {result['answers'][0]}")
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Main endpoint test failed: {e}")

if __name__ == "__main__":
    test_api()
