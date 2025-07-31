#!/usr/bin/env python3
"""
Test the LangChain-based Q&A system with the policy document
"""
import requests
import json
import time

def test_policy_document():
    base_url = "https://hackrx-bajaj.vercel.app"
    bearer_token = "ca6914a6c8df9d1ce075149c3ab9f060e666c75940576e37a98b3cf0e9092c72"
    
    print("Testing Optimized Policy Document Q&A System")
    print("=" * 70)
    
    # Your specific test case
    document_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    
    questions = [
        "Who is the prime minister of INDIA",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?",
        "What is the No Claim Discount (NCD) offered in this policy?",
        "Is there a benefit for preventive health check-ups?",
        "How does the policy define a 'Hospital'?",
        "What is the extent of coverage for AYUSH treatments?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]
    
    print(f"📄 Document: Policy PDF")
    print(f"❓ Questions: {len(questions)}")
    print("⏳ Processing with optimized system...")
    
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {bearer_token}"
        }
        
        payload = {
            "documents": document_url,
            "questions": questions
        }
        
        response = requests.post(
            f"{base_url}/hackrx/run", 
            headers=headers, 
            json=payload, 
            timeout=90  # Reasonable timeout for optimized processing
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            print("\n" + "=" * 70)
            
            for i, (question, answer) in enumerate(zip(questions, result['answers'])):
                print(f"\n❓ Q{i+1}: {question}")
                print(f"💬 A{i+1}: {answer}")
                
                # Analyze answer quality
                if "based on general knowledge" in answer.lower():
                    print("   🧠 Used general knowledge")
                elif "not available" in answer.lower() or "cannot find" in answer.lower():
                    print("   ⚠️  Not found in document")
                elif len(answer) > 30:
                    print("   📖 Document-based answer")
                else:
                    print("   ❔ Short response")
                    
            print("\n" + "=" * 70)
            print("🎯 EXPECTED RESULTS:")
            print("✅ Question 1: Should use general knowledge (PM of India)")
            print("✅ Questions 2-10: Should find specific answers in policy document")
            print("✅ All answers should be accurate and specific")
            print("=" * 70)
            
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    # Wait for deployment
    print("⏳ Waiting for optimized deployment to complete...")
    time.sleep(20)
    test_policy_document()
