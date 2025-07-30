# 🚀 VERCEL DEPLOYMENT GUIDE

## Quick Deployment Steps

### 1. **Push to GitHub**
```bash
git add .
git commit -m "🎯 Ready for Vercel deployment - Clean API for hackathon submission"
git push origin main
```

### 2. **Vercel Deployment**
Since you've already connected Vercel to GitHub:

1. **Go to Vercel Dashboard** → [vercel.com/dashboard](https://vercel.com/dashboard)
2. **Import Project** → Select your `hackrx_bajaj` repository
3. **Configure Project**:
   - Framework Preset: **Other**
   - Root Directory: **/** (leave empty)
   - Build Command: **Leave empty**
   - Output Directory: **Leave empty**
4. **Add Environment Variables** (Optional - for better performance):
   ```
   OPENAI_API_KEY=your_openai_key_here
   ANTHROPIC_API_KEY=your_anthropic_key_here
   HUGGINGFACE_API_KEY=your_huggingface_key_here
   ```
5. **Deploy** → Click "Deploy"

### 3. **Your API Endpoints**
After deployment, your API will be available at:
```
https://your-project-name.vercel.app/hackrx/run
https://your-project-name.vercel.app/health
https://your-project-name.vercel.app/
```

## 🎯 SUBMISSION ENDPOINT

**Your hackathon submission URL will be:**
```
https://your-project-name.vercel.app/hackrx/run
```

## 📋 API SPECIFICATION

### **Authentication**
```
Authorization: Bearer ca6914a6c8df9d1ce075149c3ab9f060e666c75940576e37a98b3cf0e9092c72
```

### **Request Format**
```json
POST /hackrx/run
Content-Type: application/json

{
  "documents": "https://example.com/document.pdf",
  "questions": [
    "What is the coverage period?",
    "What are the exclusions?",
    "What is the deductible amount?"
  ]
}
```

### **Response Format**
```json
{
  "answers": [
    "The coverage period is 12 months from the policy start date.",
    "Exclusions include pre-existing conditions and experimental treatments.",
    "The deductible amount is $500 per year."
  ]
}
```

## 🧪 Test Your Deployment

Use this curl command to test:

```bash
curl -X POST "https://your-project-name.vercel.app/hackrx/run" \
  -H "Authorization: Bearer ca6914a6c8df9d1ce075149c3ab9f060e666c75940576e37a98b3cf0e9092c72" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/sample-document.pdf",
    "questions": [
      "What is the main topic?",
      "What are the key points?"
    ]
  }'
```

## ✅ DEPLOYMENT CHECKLIST

- [x] ✅ Clean API with no scoring/evaluation components
- [x] ✅ Simple answers-only response format
- [x] ✅ LangChain integration for advanced processing
- [x] ✅ Multi-LLM support with robust fallbacks
- [x] ✅ PDF and text document support
- [x] ✅ Vercel configuration optimized
- [x] ✅ Bearer token authentication
- [x] ✅ Error handling and timeouts
- [ ] 🚀 Push to GitHub
- [ ] 🚀 Deploy on Vercel
- [ ] 🧪 Test deployment
- [ ] 📝 Submit endpoint URL to hackathon authority

## 🎉 YOU'RE READY!

Your API is fully optimized for hackathon submission:
- **Focus**: Accurate answer generation only
- **Evaluation**: Handled entirely by authority
- **Performance**: Advanced LangChain + Multi-LLM architecture
- **Reliability**: Comprehensive fallback systems
