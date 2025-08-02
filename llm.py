from openai import OpenAI
import os
from typing import List, Dict
from dotenv import load_dotenv
import time

load_dotenv()

class TokenTracker:
    """Track token usage for optimization monitoring"""
    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_requests = 0
        self.start_time = time.time()
    
    def add_usage(self, prompt_tokens: int, completion_tokens: int):
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_requests += 1
    
    def get_stats(self) -> Dict:
        elapsed_time = time.time() - self.start_time
        return {
            'total_prompt_tokens': self.total_prompt_tokens,
            'total_completion_tokens': self.total_completion_tokens,
            'total_tokens': self.total_prompt_tokens + self.total_completion_tokens,
            'total_requests': self.total_requests,
            'avg_tokens_per_request': (self.total_prompt_tokens + self.total_completion_tokens) / max(1, self.total_requests),
            'elapsed_time_minutes': elapsed_time / 60
        }

# Global token tracker
token_tracker = TokenTracker()

class LLMService:
    """Handle GPT-4 interactions for answer generation with token optimization"""
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # Use cheaper, faster model for better cost efficiency
        self.model = "gpt-3.5-turbo"  # Much cheaper than GPT-4, still highly accurate
        self.max_context_tokens = 1500  # Limit context size
        self.answer_cache = {}  # Simple cache for identical questions
    
    def generate_answer(self, question: str, relevant_chunks: List[Dict], document_context: str = "") -> Dict[str, str]:
        """Generate answer using optimized token usage"""
        try:
            # Check cache first
            cache_key = f"{question}_{hash(str(relevant_chunks))}"
            if cache_key in self.answer_cache:
                return self.answer_cache[cache_key]
            
            # Optimize context selection - use only top 3 most relevant chunks
            if len(relevant_chunks) > 3:
                relevant_chunks = sorted(relevant_chunks, 
                                       key=lambda x: x.get('similarity_score', 0), 
                                       reverse=True)[:3]
            
            # Prepare optimized context with token limiting
            context_parts = []
            total_tokens = 0
            
            for i, chunk in enumerate(relevant_chunks):
                chunk_text = chunk['text']
                # Estimate tokens (rough: 1 token ≈ 4 characters)
                chunk_tokens = len(chunk_text) // 4
                
                if total_tokens + chunk_tokens > self.max_context_tokens:
                    # Truncate chunk to fit within limit
                    remaining_chars = (self.max_context_tokens - total_tokens) * 4
                    chunk_text = chunk_text[:remaining_chars] + "..."
                    
                context_parts.append(f"[{i+1}] {chunk_text}")
                total_tokens += len(chunk_text) // 4
                
                if total_tokens >= self.max_context_tokens:
                    break
            
            context = " ".join(context_parts)
            
            # Ultra-compressed system prompt (saves ~100 tokens)
            system_prompt = """Expert AI: Answer insurance/legal questions in 20-30 words max. Use abbreviations, include only key facts (numbers, periods, limits). State "Not in document" if unavailable."""

            # Minimal user prompt (saves ~50 tokens)
            user_prompt = f"Q: {question}\nContext: {context}\nAnswer (20-30 words):"
            
            # Generate response with token optimization and tracking
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=35  # Strict limit for cost control
            )
            
            # Track token usage
            if hasattr(response, 'usage') and response.usage:
                token_tracker.add_usage(
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens
                )
            
            response_text = response.choices[0].message.content.strip()
            
            # Cache the result
            result = {
                'answer': response_text,
                'clause': '',
                'rationale': ''
            }
            self.answer_cache[cache_key] = result
            
            return result
        except Exception as e:
            return {
                'answer': f"Error: {str(e)[:50]}...",  # Truncate error messages
                'clause': "",
                'rationale': ""
            }
        """Get current token usage statistics"""
        return token_tracker.get_stats()
    
    def clear_cache(self):
        """Clear the answer cache"""
        self.answer_cache.clear()
        
    def preprocess_question(self, question: str) -> str:
        """Preprocess question to standardize format and save tokens"""
        # Remove unnecessary words and standardize
        question = question.strip().lower()
        
        # Replace verbose phrases with shorter equivalents
        replacements = {
            'what is the': 'what\'s the',
            'please tell me': '',
            'can you explain': '',
            'i would like to know': '',
            'could you please': '',
            'under this policy': '',
            'in this document': '',
        }
        
        for old, new in replacements.items():
            question = question.replace(old, new)
        
        return question.strip()
    
    def _parse_response(self, response_text: str) -> Dict[str, str]:
        """Parse the structured response from GPT-4"""
        result = {
            'answer': '',
            'clause': '',
            'rationale': ''
        }
        
        try:
            lines = response_text.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                
                if line.startswith('ANSWER:'):
                    current_section = 'answer'
                    result['answer'] = line[7:].strip()
                elif line.startswith('CLAUSE:'):
                    current_section = 'clause'
                    result['clause'] = line[7:].strip()
                elif line.startswith('RATIONALE:'):
                    current_section = 'rationale'
                    result['rationale'] = line[10:].strip()
                elif current_section and line:
                    # Continue adding to current section
                    if result[current_section]:
                        result[current_section] += ' ' + line
                    else:
                        result[current_section] = line
            
            # Fallback: if parsing fails, put everything in answer
            if not result['answer']:
                result['answer'] = response_text
                result['clause'] = "See answer above"
                result['rationale'] = "Response format could not be parsed"
            
        except Exception as e:
            result = {
                'answer': response_text,
                'clause': "Parsing error",
                'rationale': f"Error parsing response: {str(e)}"
            }
        
        return result
    
    def generate_summary(self, text: str, max_length: int = 200) -> str:
        """Generate a summary of the document"""
        try:
            prompt = f"""Please provide a concise summary of the following document in {max_length} words or less:

{text[:3000]}  # Limit input to avoid token limits

Focus on:
- Document type (insurance policy, legal contract, etc.)
- Key provisions or benefits
- Important terms and conditions
- Coverage areas or scope"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Summary generation failed: {str(e)}"
    
    def extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from the document"""
        try:
            prompt = f"""Analyze the following document and extract the most important key terms, concepts, and defined terms. Return them as a comma-separated list:

{text[:2000]}

Focus on:
- Legal or insurance terminology
- Defined terms
- Important concepts
- Coverage types
- Key provisions"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=200
            )
            
            terms_text = response.choices[0].message.content
            # Split by commas and clean up
            terms = [term.strip() for term in terms_text.split(',')]
            return [term for term in terms if term]
            
        except Exception as e:
            return [f"Error extracting terms: {str(e)}"]

# Convenience functions
def answer_question(question: str, relevant_chunks: List[Dict]) -> Dict[str, str]:
    """Generate answer for a question using relevant chunks"""
    service = LLMService()
    return service.generate_answer(question, relevant_chunks)

def summarize_document(text: str) -> str:
    """Generate document summary"""
    service = LLMService()
    return service.generate_summary(text)
