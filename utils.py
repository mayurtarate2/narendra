import hashlib
import json
import time
from typing import Any, Dict, List, Optional
import re

def generate_document_id(url_or_content: str) -> str:
    """Generate a unique document ID based on content or URL"""
    content_hash = hashlib.md5(url_or_content.encode()).hexdigest()
    timestamp = str(int(time.time()))
    return f"doc_{content_hash[:8]}_{timestamp}"

def clean_text(text: str) -> str:
    """Clean text for processing"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters that might interfere with processing
    text = re.sub(r'[^\w\s.,;:!?()-]', ' ', text)
    
    # Remove excessive punctuation
    text = re.sub(r'[.]{3,}', '...', text)
    
    return text.strip()

def estimate_tokens(text: str) -> int:
    """Rough estimation of token count for text"""
    # Rough approximation: 1 token ≈ 0.75 words
    words = len(text.split())
    return int(words * 1.3)

def truncate_text(text: str, max_tokens: int = 3000) -> str:
    """Truncate text to approximate token limit"""
    estimated_tokens = estimate_tokens(text)
    
    if estimated_tokens <= max_tokens:
        return text
    
    # Calculate approximate characters to keep
    chars_per_token = len(text) / estimated_tokens
    max_chars = int(max_tokens * chars_per_token)
    
    return text[:max_chars] + "..."

def format_processing_time(start_time: float) -> float:
    """Calculate and format processing time"""
    return round(time.time() - start_time, 2)

def validate_url(url: str) -> bool:
    """Validate if string is a valid URL"""
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    return url_pattern.match(url) is not None

def extract_file_type(url_or_content: str) -> str:
    """Extract file type from URL or content"""
    if validate_url(url_or_content):
        if '.pdf' in url_or_content.lower():
            return 'pdf'
        elif '.docx' in url_or_content.lower() or '.doc' in url_or_content.lower():
            return 'docx'
        else:
            return 'unknown'
    else:
        # Check if it looks like email content
        if any(marker in url_or_content for marker in ['From:', 'Subject:', 'To:', '@']):
            return 'email'
        else:
            return 'text'

def safe_json_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure JSON response is serializable"""
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    return make_serializable(data)

def chunk_text_simple(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    """Simple text chunking fallback if spaCy/NLTK fails"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = ' '.join(chunk_words)
        chunks.append(chunk_text)
        
        if i + chunk_size >= len(words):
            break
    
    return chunks

def normalize_question(question: str) -> str:
    """Normalize question text for better processing"""
    # Convert to lowercase and remove extra whitespace
    question = re.sub(r'\s+', ' ', question.strip().lower())
    
    # Ensure question ends with question mark
    if not question.endswith('?'):
        question += '?'
    
    return question

def extract_keywords(text: str) -> List[str]:
    """Extract potential keywords from text"""
    # Remove common stop words and extract meaningful terms
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
        'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'over', 'under', 'again', 'further',
        'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
        'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
        'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
        'can', 'will', 'just', 'should', 'now', 'is', 'are', 'was', 'were',
        'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'would'
    }
    
    # Extract words and filter
    words = re.findall(r'\b\w+\b', text.lower())
    keywords = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Get unique keywords, preserving order
    unique_keywords = []
    seen = set()
    for keyword in keywords:
        if keyword not in seen:
            unique_keywords.append(keyword)
            seen.add(keyword)
    
    return unique_keywords[:20]  # Return top 20 keywords

def log_performance(operation: str, duration: float, details: Dict[str, Any] = None):
    """Log performance metrics"""
    log_entry = {
        'timestamp': time.time(),
        'operation': operation,
        'duration': duration,
        'details': details or {}
    }
    
    # In a production system, this would go to a proper logging service
    print(f"[PERF] {operation}: {duration:.2f}s {details or ''}")

def retry_operation(func, max_retries: int = 3, delay: float = 1.0):
    """Retry an operation with exponential backoff"""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            
            wait_time = delay * (2 ** attempt)
            time.sleep(wait_time)
    
    return None

def sanitize_filename(text: str, max_length: int = 50) -> str:
    """Sanitize text to be used as filename"""
    # Remove invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', text)
    
    # Remove extra spaces and limit length
    sanitized = re.sub(r'\s+', '_', sanitized).strip('_')
    
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rstrip('_')
    
    return sanitized or 'untitled'
