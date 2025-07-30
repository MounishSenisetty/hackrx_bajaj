#!/bin/bash

# Curl test script for the HackRX API
# This script tests the API endpoints using curl

set -e  # Exit on any error

echo "🧪 Testing HackRX API with curl"
echo "================================="

# Configuration
BEARER_TOKEN="ca6914a6c8df9d1ce075149c3ab9f060e666c75940576e37a98b3cf0e9092c72"
LOCAL_URL="http://localhost:8000"
VERCEL_URL=""  # Add your Vercel URL here when deployed

# Test payload
TEST_PAYLOAD='{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment?",
        "What is the waiting period for pre-existing diseases?"
    ]
}'

# Function to test an endpoint
test_endpoint() {
    local base_url=$1
    local endpoint=$2
    local method=$3
    local data=$4
    local description=$5
    
    echo ""
    echo "📡 Testing: $description"
    echo "URL: $base_url$endpoint"
    echo "Method: $method"
    
    if [ "$method" = "GET" ]; then
        response=$(curl -s -w "\nHTTP_STATUS:%{http_code}" "$base_url$endpoint" || echo "ERROR")
    else
        response=$(curl -s -w "\nHTTP_STATUS:%{http_code}" \
            -X "$method" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer $BEARER_TOKEN" \
            -d "$data" \
            "$base_url$endpoint" || echo "ERROR")
    fi
    
    if [[ "$response" == *"ERROR"* ]]; then
        echo "❌ Connection failed"
        return 1
    fi
    
    http_status=$(echo "$response" | grep "HTTP_STATUS" | cut -d: -f2)
    response_body=$(echo "$response" | sed '/HTTP_STATUS/d')
    
    echo "Status: $http_status"
    
    if [ "$http_status" -eq 200 ]; then
        echo "✅ Success"
        echo "Response preview:"
        echo "$response_body" | head -n 10
    elif [ "$http_status" -eq 401 ]; then
        echo "🔐 Authentication required"
    elif [ "$http_status" -eq 403 ]; then
        echo "🚫 Forbidden - Invalid token"
    else
        echo "⚠️  Unexpected status: $http_status"
        echo "Response: $response_body"
    fi
    
    return 0
}

# Function to test a base URL
test_api_base() {
    local base_url=$1
    local name=$2
    
    echo ""
    echo "🌐 Testing $name API at: $base_url"
    echo "================================================"
    
    # Test health endpoint
    test_endpoint "$base_url" "/health" "GET" "" "Health Check"
    
    # Test root endpoint
    test_endpoint "$base_url" "/" "GET" "" "Root Endpoint"
    
    # Test system info
    test_endpoint "$base_url" "/system/info" "GET" "" "System Info"
    
    # Test main API endpoint
    test_endpoint "$base_url" "/hackrx/run" "POST" "$TEST_PAYLOAD" "Main API Endpoint"
    
    echo ""
    echo "✨ Completed testing $name"
}

# Check if local server is running
echo "🔍 Checking if local server is running..."
if curl -s "$LOCAL_URL/health" > /dev/null 2>&1; then
    echo "✅ Local server detected at $LOCAL_URL"
    LOCAL_AVAILABLE=true
else
    echo "❌ Local server not running at $LOCAL_URL"
    LOCAL_AVAILABLE=false
fi

# Test local API if available
if [ "$LOCAL_AVAILABLE" = true ]; then
    test_api_base "$LOCAL_URL" "Local"
else
    echo ""
    echo "💡 To test locally, run:"
    echo "   python3 main_simple.py"
    echo "   # or"
    echo "   uvicorn main:app --host 0.0.0.0 --port 8000"
fi

# Test Vercel deployment if URL is provided
if [ -n "$VERCEL_URL" ]; then
    test_api_base "$VERCEL_URL" "Vercel"
else
    echo ""
    echo "💡 To test Vercel deployment:"
    echo "   1. Deploy to Vercel: vercel --prod"
    echo "   2. Update VERCEL_URL in this script"
    echo "   3. Run this script again"
fi

echo ""
echo "🎯 Quick curl command for manual testing:"
echo ""
echo "curl -X POST '$LOCAL_URL/hackrx/run' \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -H 'Authorization: Bearer $BEARER_TOKEN' \\"
echo "  -d '$TEST_PAYLOAD'"

echo ""
echo "🚀 Testing completed!"
