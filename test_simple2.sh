#!/bin/bash

echo "Testing with a simple text document..."

curl -s \
  -X POST \
  "https://hackrx-bajaj.vercel.app/hackrx/run" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -H "Authorization: Bearer ca6914a6c8df9d1ce075149c3ab9f060e666c75940576e37a98b3cf0e9092c72" \
  -d '{
    "documents": "https://www.example.com",
    "questions": ["What is this about?"]
  }' | jq .

echo -e "\nTesting with the original policy PDF URL..."

curl -s \
  -X POST \
  "https://hackrx-bajaj.vercel.app/hackrx/run" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -H "Authorization: Bearer ca6914a6c8df9d1ce075149c3ab9f060e666c75940576e37a98b3cf0e9092c72" \
  -d '{
    "documents": "https://cdn.getmoto.com/sample-documents/policy.pdf",
    "questions": ["What is the grace period for premium payment?"]
  }' | jq .
