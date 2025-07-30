#!/bin/bash

echo "Testing Generic PDF Processing with Different Document Types"
echo "========================================================"

# Test 1: Original policy document
echo -e "\n1. Testing with Insurance Policy Document:"
curl -s -X POST "https://hackrx-bajaj.vercel.app/hackrx/run" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ca6914a6c8df9d1ce075149c3ab9f060e666c75940576e37a98b3cf0e9092c72" \
  -d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": ["What is the grace period for premium payment?"]
  }' | jq -r '.answers[0]'

# Test 2: Simple text document (simulating different PDF type)
echo -e "\n2. Testing with Simple Text Content:"
curl -s -X POST "https://hackrx-bajaj.vercel.app/hackrx/run" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ca6914a6c8df9d1ce075149c3ab9f060e666c75940576e37a98b3cf0e9092c72" \
  -d '{
    "documents": "https://httpbin.org/json",
    "questions": ["What is the purpose of this service?"]
  }' | jq -r '.answers[0]'

# Test 3: Research paper style questions
echo -e "\n3. Testing with Research-Style Questions on Policy Doc:"
curl -s -X POST "https://hackrx-bajaj.vercel.app/hackrx/run" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ca6914a6c8df9d1ce075149c3ab9f060e666c75940576e37a98b3cf0e9092c72" \
  -d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
      "What methodology is used in this document?",
      "How much does it cost?",
      "When was this document created?",
      "What are the main findings?",
      "Is there any statistical data mentioned?"
    ]
  }' | jq -r '.answers[] | "• " + .'

echo -e "\n========================================================"
echo "Generic PDF Processing Test Complete!"
