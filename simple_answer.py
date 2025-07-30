"""
Simple answer generation without external LLM API calls
This module provides fallback functionality to avoid 403 errors
"""

import re
from typing import List, Dict, Any
from datetime import datetime

class SimpleAnswerGenerator:
    """Generate answers using simple text processing without external APIs."""
    
    @staticmethod
    def extract_relevant_sentence(question: str, context: str) -> str:
        """Extract the most relevant sentence from context."""
        question_keywords = set(re.findall(r'\w+', question.lower()))
        sentences = re.split(r'[.!?]+', context)
        
        best_sentence = ""
        best_score = 0
        
        for sentence in sentences:
            if len(sentence.strip()) < 10:
                continue
                
            sentence_words = set(re.findall(r'\w+', sentence.lower()))
            overlap = len(question_keywords.intersection(sentence_words))
            
            if overlap > best_score:
                best_score = overlap
                best_sentence = sentence.strip()
        
        return best_sentence if best_score > 0 else ""
    
    @staticmethod
    def generate_answer(question: str, relevant_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate answer from relevant chunks without external API calls."""
        if not relevant_chunks:
            return {
                "answer": "Information not available in the provided document.",
                "confidence_score": 0.0,
                "relevant_chunks": [],
                "reasoning": "No relevant chunks found for the question."
            }
        
        # Get the best chunk
        best_chunk = relevant_chunks[0]
        context = best_chunk['text']
        
        # Try to extract a relevant sentence
        relevant_sentence = SimpleAnswerGenerator.extract_relevant_sentence(question, context)
        
        if relevant_sentence:
            answer = relevant_sentence
            if not answer.endswith('.'):
                answer += '.'
        else:
            # Fallback to first part of the chunk
            answer = context[:400].strip()
            if len(context) > 400:
                answer += "..."
        
        confidence = min(best_chunk.get('score', 0.5), 1.0)
        
        return {
            "answer": answer,
            "confidence_score": confidence,
            "relevant_chunks": relevant_chunks,
            "reasoning": f"Answer extracted from {len(relevant_chunks)} relevant document sections using text analysis (no external API calls)."
        }
