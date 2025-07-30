#!/bin/bash

echo "🚀 TESTING DEPLOYED NATURAL LLM SYSTEM"
echo "======================================"
echo ""
echo "Testing API: https://hackrx-bajaj.vercel.app/hackrx/run"
echo "Expected: Natural answers extracted from document context"
echo "No hardcoded patterns should appear in responses"
echo ""

# Test the deployed API with the exact payload from user requirements
curl -X POST "https://hackrx-bajaj.vercel.app/hackrx/run" \
  -H "Authorization: Bearer ca6914a6c8df9d1ce075149c3ab9f060e666c75940576e37a98b3cf0e9092c72" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?",
        "What is the No Claim Discount (NCD) offered in this policy?",
        "Is there a benefit for preventive health check-ups?",
        "How does the policy define a '\''Hospital'\''?",
        "What is the extent of coverage for AYUSH treatments?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]
}' \
  --max-time 120 \
  --connect-timeout 30 \
  -w "\n\n⏱️  Response Time: %{time_total} seconds\n📊 Status Code: %{http_code}\n" | jq '.'

echo ""
echo "✅ TEST COMPLETE!"
echo ""
echo "🔍 VALIDATION CHECKLIST:"
echo "========================"
echo "✅ Check if answers contain specific numbers from document (30 days, 36 months, etc.)"
echo "✅ Verify responses are contextual and not generic"
echo "✅ Ensure no hardcoded patterns like 'The time period is 90 days'"
echo "✅ Look for natural language extracted from actual policy text"
echo "✅ Response time should be under 10 seconds for good performance"
echo ""
echo "If answers look natural and contain document-specific details,"
echo "then the natural LLM system is working correctly! 🎉"
