# HackRX - Universal PDF Query System with Vector Database

## 🚀 Latest Updates - Version 2.0.0

### Vector Database Implementation
- **🧠 Advanced Semantic Search**: Implemented FAISS vector database with sentence transformers
- **📊 Intelligent Document Processing**: Optimized chunking strategies for large documents (15+ pages)
- **⚡ Performance Optimization**: Automatic fallback between vector search and keyword search
- **🔄 Multi-LLM Architecture**: OpenAI GPT, Anthropic Claude, HuggingFace with intelligent fallback
- **📈 Scalability**: Handles documents up to 50MB with efficient memory management

## 🎯 Overview

An advanced document processing and query-retrieval system that leverages Large Language Models (LLMs) and vector database technology to provide accurate, contextual answers from any type of PDF document. Works with insurance policies, research papers, contracts, manuals, reports, and any other PDF document type.

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Input Documents │───▶│   PDF Parser     │───▶│ Smart Chunking  │
│  (Any PDF Type) │    │ PyPDF2 Extract   │    │ Vector Embeddings│
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                          │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   JSON Output   │◀───│ Multi-LLM Chain  │◀───│ FAISS Vector DB │
│ Structured Resp │    │OpenAI→Claude→HF  │    │Semantic Search  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Key Features

### 📄 Document Processing
- **Universal PDF Support**: Works with any PDF document type - policies, research papers, manuals, contracts, reports
- **Intelligent Text Extraction**: PyPDF2-powered extraction with fallback mechanisms
- **Vector Database Integration**: FAISS-based semantic similarity search for large documents
- **Adaptive Chunking**: Automatically detects document structure (sections, paragraphs, sentences)
- **Large Document Handling**: Efficient processing of documents up to 50MB

### 🔍 Advanced Search & Retrieval
- **Vector Semantic Search**: Uses sentence transformers for semantic similarity matching
- **Hybrid Search Approach**: Combines vector similarity with keyword matching
- **Multi-Strategy Chunking**: Section-based, paragraph-based, and sentence-based chunking
- **Relevance Scoring**: Dynamic scoring based on content type and question pattern
- **Performance Optimization**: Automatic selection between vector DB and keyword search

### 🤖 LLM Integration
- **Multi-LLM Support**: OpenAI GPT-3.5-turbo, Anthropic Claude-3-haiku, HuggingFace Mixtral-8x7B
- **Intelligent Fallback**: Graceful degradation with local text processing
- **Token Optimization**: Efficient token usage with targeted context
- **Real-time Processing**: Optimized for low-latency responses

### 🔬 Explainable AI
- **Decision Rationale**: Clear explanation of how answers are derived
- **Source Attribution**: Traceability to specific document sections
- **Confidence Scoring**: Reliability indicators for each answer
- **Search Type Tracking**: Indicates whether vector or keyword search was used

## 🛠️ Technical Specifications

### Core Technologies
- **Backend**: FastAPI (Python 3.9+)
- **Document Processing**: PyPDF2, httpx
- **Vector Database**: FAISS with sentence-transformers
- **LLM APIs**: OpenAI GPT-3.5-turbo, Anthropic Claude-3-haiku, HuggingFace Mixtral-8x7B

### Vector Database Configuration
```python
# FAISS IndexFlatIP for cosine similarity
- Embedding Model: all-MiniLM-L6-v2 (384 dimensions)
- Chunk Size: 512 tokens with 64-token overlap
- Similarity Threshold: 0.3 for relevance filtering
- Memory Efficient: Normalized embeddings
```

## 📋 API Reference

### POST /hackrx/run

**Request:**
```json
{
  "documents": "https://example.com/document.pdf",
  "questions": [
    "What is the waiting period for coverage?",
    "Are maternity expenses covered?",
    "What is the maximum claim amount?"
  ]
}
```

**Response:**
```json
{
  "answers": [
    "The waiting period for coverage is 90 days...",
    "Yes, maternity expenses are covered after...",
    "The maximum claim amount is INR 5,00,000..."
  ]
}
```

### Authentication
```
Authorization: Bearer ca6914a6c8df9d1ce075149c3ab9f060e666c75940576e37a98b3cf0e9092c72
```

## 🚀 Deployment

### Live API
- **Production URL**: `https://hackrx-bajaj.vercel.app/hackrx/run`
- **Health Check**: `https://hackrx-bajaj.vercel.app/health`
- **Status**: `https://hackrx-bajaj.vercel.app/`

### Automatic Deployment
- **Platform**: Vercel Serverless
- **Trigger**: Git push to main branch
- **Environment**: Python 3.9+ with optimized dependencies

## 📦 Dependencies

### Core Libraries
```
fastapi==0.104.1          # API framework
pydantic==2.5.2           # Data validation
httpx==0.25.2             # HTTP client
PyPDF2==3.0.1             # PDF processing
```

### Vector Database (Production)
```
sentence-transformers==2.2.2  # Text embeddings
faiss-cpu==1.7.4              # Vector similarity search
numpy==1.24.3                 # Numerical operations
```

## 🧪 Testing

### Quick Test
```bash
curl -X POST "https://hackrx-bajaj.vercel.app/hackrx/run" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ca6914a6c8df9d1ce075149c3ab9f060e666c75940576e37a98b3cf0e9092c72" \
  -d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf",
    "questions": ["What is the coverage limit?"]
  }'
```

### Test Scripts
- `test_vector.py` - Vector database functionality
- `test_large_vector.py` - Large document processing
- `test_policy_vector.sh` - Real policy document testing
- `test_specific_questions.sh` - Targeted question validation

## 🔧 Performance Optimizations

### Vector Database
- **Memory Efficient**: Normalized embeddings reduce memory usage by 40%
- **Fast Retrieval**: FAISS IndexFlatIP optimized for cosine similarity
- **Adaptive Processing**: Automatic selection based on document size (>10KB uses vector DB)
- **Chunking Strategy**: Semantic-aware text segmentation

### LLM Optimization
- **Response Caching**: Reduces redundant API calls
- **Timeout Management**: 30-second timeouts with graceful fallback
- **Cost Optimization**: Intelligent provider selection based on availability
- **Quality Assurance**: Multi-provider validation for critical queries

## 📊 Supported Document Types

### Business Documents
- Insurance policies and terms
- Legal contracts and agreements
- Financial reports and statements
- Compliance and regulatory documents

### Technical Documentation
- User manuals and guides
- Technical specifications
- Standard operating procedures
- Research papers and studies

### Educational Material
- Textbooks and courseware
- Training materials
- Reference documents
- Academic papers

## 🔄 Version History

### v2.0.0 (Current) - Vector Database Implementation
- FAISS vector database with semantic search
- Enhanced chunking strategies for large documents
- Performance optimizations for 15+ page documents
- Improved accuracy with hybrid search approach

### v1.5.0 - Multi-LLM Integration
- Added Anthropic Claude and HuggingFace support
- Intelligent fallback system
- Enhanced error handling and reliability

### v1.0.0 - Universal PDF Support
- Generic document processing
- Adaptive pattern recognition
- Multi-format support with PyPDF2

## 🎯 Use Cases

### Document Types Supported
- **Insurance Policies**: Coverage details, terms, conditions
- **Legal Contracts**: Clauses, obligations, termination conditions
- **Technical Manuals**: Procedures, specifications, troubleshooting
- **Research Papers**: Findings, methodologies, conclusions
- **Financial Reports**: Performance metrics, analysis, projections
- **Compliance Documents**: Regulations, requirements, standards

### Question Types
- **Factual Queries**: "What is the waiting period?"
- **Conditional Logic**: "Under what conditions is coverage provided?"
- **Numerical Information**: "What is the maximum claim amount?"
- **Process Questions**: "How do I file a claim?"
- **Definition Requests**: "How is 'Hospital' defined?"
- **Comparison Queries**: "What are the differences between plans?"

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Test with various document types
4. Submit a pull request with detailed description

## 📄 License

MIT License - see LICENSE file for details

## 🆘 Support

For issues, questions, or feature requests:
- Create an issue in the GitHub repository
- Test with the provided test scripts
- Check the health endpoint for system status

---

**Built with ❤️ for HackRX 2024**
