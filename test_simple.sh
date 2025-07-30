#!/bin/bash

echo "Testing with simple text document..."

curl -s -X POST "https://hackrx-bajaj.vercel.app/hackrx/run" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -H "Authorization: Bearer ca6914a6c8df9d1ce075149c3ab9f060e666c75940576e37a98b3cf0e9092c72" \
  -d '{
    "documents": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
    "questions": [
        "What is this document about?"
    ]
  }'
