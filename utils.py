"""
Utility functions for the LLM-Powered Intelligent Query-Retrieval System
"""

import re
import hashlib
import tiktoken
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class TextUtils:
    """Utility functions for text processing."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text content."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.,!?;:()\-"\']', ' ', text)
        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)
        return text.strip()
    
    @staticmethod
    def extract_key_phrases(text: str) -> List[str]:
        """Extract key phrases from text."""
        # Simple keyword extraction (can be enhanced with NLP libraries)
        words = text.lower().split()
        # Filter out common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did'
        }
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]
        return list(set(keywords))
    
    @staticmethod
    def count_tokens(text: str, model: str = "gpt-4") -> int:
        """Count tokens in text using tiktoken."""
        try:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except Exception:
            # Fallback approximation
            return len(text.split()) * 1.3

class DocumentUtils:
    """Utility functions for document processing."""
    
    @staticmethod
    def generate_document_id(url: str) -> str:
        """Generate unique document ID from URL."""
        return hashlib.md5(url.encode()).hexdigest()
    
    @staticmethod
    def validate_document_size(content: bytes, max_size: int = 50 * 1024 * 1024) -> bool:
        """Validate document size."""
        return len(content) <= max_size
    
    @staticmethod
    def detect_language(text: str) -> str:
        """Simple language detection (can be enhanced with language detection libraries)."""
        # Basic heuristic - check for common English words
        english_words = {'the', 'and', 'of', 'to', 'a', 'in', 'for', 'is', 'on', 'that', 'by', 'this', 'with', 'i', 'you', 'it'}
        words = set(text.lower().split()[:100])  # Check first 100 words
        english_count = len(words.intersection(english_words))
        
        if english_count >= 5:
            return 'en'
        else:
            return 'unknown'

class PerformanceUtils:
    """Utility functions for performance monitoring."""
    
    @staticmethod
    def calculate_processing_stats(
        start_time: float,
        end_time: float,
        document_length: int,
        chunk_count: int,
        question_count: int
    ) -> Dict[str, Any]:
        """Calculate processing performance statistics."""
        processing_time = end_time - start_time
        
        return {
            'total_processing_time': round(processing_time, 3),
            'time_per_question': round(processing_time / question_count, 3) if question_count > 0 else 0,
            'characters_per_second': round(document_length / processing_time, 2) if processing_time > 0 else 0,
            'chunks_per_second': round(chunk_count / processing_time, 2) if processing_time > 0 else 0,
            'efficiency_score': min(100, round((document_length / 1000) / max(processing_time, 0.1), 2))
        }

class ValidationUtils:
    """Utility functions for input validation."""
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate URL format."""
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        return url_pattern.match(url) is not None
    
    @staticmethod
    def validate_questions(questions: List[str]) -> List[str]:
        """Validate and clean questions."""
        valid_questions = []
        for q in questions:
            if isinstance(q, str) and len(q.strip()) > 0:
                # Clean the question
                cleaned = TextUtils.clean_text(q.strip())
                if len(cleaned) >= 5:  # Minimum question length
                    valid_questions.append(cleaned)
        return valid_questions
    
    @staticmethod
    def sanitize_input(text: str) -> str:
        """Sanitize user input to prevent injection attacks."""
        # Remove potential script tags and dangerous characters
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'[<>]', '', text)
        return text.strip()

class ResponseUtils:
    """Utility functions for response formatting."""
    
    @staticmethod
    def format_confidence_score(score: float) -> str:
        """Format confidence score as human-readable string."""
        if score >= 0.9:
            return "Very High"
        elif score >= 0.7:
            return "High"
        elif score >= 0.5:
            return "Medium"
        elif score >= 0.3:
            return "Low"
        else:
            return "Very Low"
    
    @staticmethod
    def generate_explanation(chunks: List[Dict[str, Any]], method: str = "semantic_search") -> str:
        """Generate explanation for how the answer was derived."""
        if not chunks:
            return "No relevant information found in the document."
        
        chunk_count = len(chunks)
        avg_score = sum(chunk['score'] for chunk in chunks) / chunk_count
        max_score = max(chunk['score'] for chunk in chunks)
        
        explanation = f"Answer derived using {method} across {chunk_count} relevant document section(s). "
        explanation += f"Average relevance score: {avg_score:.3f}, highest score: {max_score:.3f}. "
        
        if max_score >= 0.8:
            explanation += "High confidence in answer accuracy based on strong semantic match."
        elif max_score >= 0.6:
            explanation += "Good confidence in answer accuracy based on semantic similarity."
        else:
            explanation += "Moderate confidence - answer may require verification."
        
        return explanation
