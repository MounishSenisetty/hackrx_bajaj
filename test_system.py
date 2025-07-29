"""
Test script for the LLM-Powered Intelligent Query-Retrieval System
"""

import asyncio
import httpx
import json
import time
from typing import Dict, Any

class SystemTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.headers = {
            "Authorization": "Bearer ca6914a6c8df9d1ce075149c3ab9f060e666c75940576e37a98b3cf0e9092c72",
            "Content-Type": "application/json"
        }
    
    async def test_health_check(self):
        """Test the health check endpoint."""
        print("🔍 Testing health check endpoint...")
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{self.base_url}/health")
                if response.status_code == 200:
                    print("✅ Health check passed")
                    print(f"Response: {response.json()}")
                else:
                    print(f"❌ Health check failed: {response.status_code}")
            except Exception as e:
                print(f"❌ Health check error: {e}")
    
    async def test_system_info(self):
        """Test the system info endpoint."""
        print("\n🔍 Testing system info endpoint...")
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{self.base_url}/system/info")
                if response.status_code == 200:
                    print("✅ System info endpoint working")
                    info = response.json()
                    print(f"System: {info.get('system_name', 'Unknown')}")
                    print(f"Capabilities: {len(info.get('capabilities', []))} features")
                else:
                    print(f"❌ System info failed: {response.status_code}")
            except Exception as e:
                print(f"❌ System info error: {e}")
    
    async def test_document_processing(self):
        """Test the main document processing endpoint."""
        print("\n🔍 Testing document processing...")
        
        test_payload = {
            "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
            "questions": [
                "What is the grace period for premium payment?",
                "What is the waiting period for pre-existing diseases?",
                "Does this policy cover maternity expenses?",
                "What is the waiting period for cataract surgery?",
                "Are medical expenses for organ donors covered?"
            ]
        }
        
        start_time = time.time()
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/hackrx/run",
                    headers=self.headers,
                    json=test_payload
                )
                
                processing_time = time.time() - start_time
                
                if response.status_code == 200:
                    print("✅ Document processing successful")
                    result = response.json()
                    
                    print(f"Processing time: {processing_time:.2f} seconds")
                    print(f"Questions processed: {len(result.get('answers', []))}")
                    
                    # Print sample answers
                    answers = result.get('answers', [])
                    for i, answer in enumerate(answers[:2]):  # Show first 2 answers
                        print(f"\nQ{i+1}: {test_payload['questions'][i]}")
                        print(f"A{i+1}: {answer[:200]}..." if len(answer) > 200 else f"A{i+1}: {answer}")
                    
                    # Print processing stats if available
                    stats = result.get('processing_stats', {})
                    if stats:
                        print(f"\nProcessing Stats:")
                        print(f"- Document chunks: {stats.get('total_chunks', 'N/A')}")
                        print(f"- Processing time: {stats.get('processing_time_seconds', 'N/A')}s")
                        print(f"- Embedding model: {stats.get('embedding_model', 'N/A')}")
                        print(f"- Vector store: {stats.get('vector_store', 'N/A')}")
                
                else:
                    print(f"❌ Document processing failed: {response.status_code}")
                    print(f"Error: {response.text}")
                    
            except Exception as e:
                print(f"❌ Document processing error: {e}")
    
    async def test_authentication(self):
        """Test authentication requirements."""
        print("\n🔍 Testing authentication...")
        
        # Test without token
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/hackrx/run",
                    json={"documents": "test", "questions": ["test"]}
                )
                if response.status_code == 401:
                    print("✅ Authentication correctly required")
                else:
                    print(f"❌ Authentication test failed: {response.status_code}")
            except Exception as e:
                print(f"❌ Authentication test error: {e}")
        
        # Test with wrong token
        wrong_headers = {
            "Authorization": "Bearer wrong_token",
            "Content-Type": "application/json"
        }
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/hackrx/run",
                    headers=wrong_headers,
                    json={"documents": "test", "questions": ["test"]}
                )
                if response.status_code == 403:
                    print("✅ Invalid token correctly rejected")
                else:
                    print(f"❌ Invalid token test failed: {response.status_code}")
            except Exception as e:
                print(f"❌ Invalid token test error: {e}")
    
    async def test_error_handling(self):
        """Test error handling for various scenarios."""
        print("\n🔍 Testing error handling...")
        
        # Test with invalid document URL
        invalid_payload = {
            "documents": "https://invalid-url-that-does-not-exist.com/fake.pdf",
            "questions": ["What is this document about?"]
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/hackrx/run",
                    headers=self.headers,
                    json=invalid_payload
                )
                if response.status_code >= 400:
                    print("✅ Invalid URL correctly handled")
                else:
                    print(f"❌ Invalid URL test unexpected result: {response.status_code}")
            except Exception as e:
                print(f"⚠️  Error handling test: {e}")
        
        # Test with empty questions
        empty_payload = {
            "documents": "https://example.com/test.pdf",
            "questions": []
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/hackrx/run",
                    headers=self.headers,
                    json=empty_payload
                )
                print(f"Empty questions test: {response.status_code}")
            except Exception as e:
                print(f"⚠️  Empty questions test error: {e}")
    
    async def run_all_tests(self):
        """Run all tests."""
        print("🚀 Starting LLM-Powered Query-Retrieval System Tests\n")
        print("=" * 60)
        
        await self.test_health_check()
        await self.test_system_info()
        await self.test_authentication()
        await self.test_error_handling()
        await self.test_document_processing()
        
        print("\n" + "=" * 60)
        print("🏁 All tests completed!")

async def main():
    """Main test function."""
    tester = SystemTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())
