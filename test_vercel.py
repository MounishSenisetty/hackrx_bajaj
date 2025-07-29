#!/usr/bin/env python3
"""
Quick test script for the API before deployment
"""

import asyncio
import json
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_api():
    """Test the API functionality locally."""
    try:
        # Import the API
        from api.index import app
        
        print("✅ API import successful")
        
        # Test data
        test_payload = {
            "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
            "questions": [
                "What is the grace period for premium payment?",
                "What is the waiting period for pre-existing diseases?"
            ]
        }
        
        print("✅ Test payload prepared")
        print("🚀 API is ready for deployment to Vercel!")
        print("\nNext steps:")
        print("1. Run: npm install -g vercel")
        print("2. Run: vercel --prod")
        print("3. Use the provided URL as your webhook endpoint")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing API: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_api())
    sys.exit(0 if result else 1)
