import spacy
import nltk
from nltk.tokenize import sent_tokenize
from typing import List, Dict
import re

class TextChunker:
    """Handle intelligent text chunking with spaCy and NLTK"""
    
    def __init__(self):
        self.nlp = None
        self._init_nlp()
        
    def _init_nlp(self):
        """Initialize spaCy model"""
        try:
            # Try to load English model
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("spaCy English model not found. Please install with: python -m spacy download en_core_web_sm")
            # Fallback to basic processing
            self.nlp = None
    
    def _download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep sentence structure
        text = re.sub(r'[^\w\s.,;:!?()-]', ' ', text)
        
        # Remove excessive newlines
        text = re.sub(r'\n+', '\n', text)
        
        return text.strip()
    
    def detect_clauses(self, text: str) -> List[str]:
        """Detect legal/insurance clauses using spaCy and regex patterns"""
        clauses = []
        
        # Common clause patterns in legal/insurance documents
        clause_patterns = [
            r'(?i)(?:section|clause|article|paragraph)\s+\d+[\.\w]*[:\-\s].*?(?=(?:section|clause|article|paragraph)\s+\d+|$)',
            r'(?i)(?:coverage|benefit|exclusion|condition|term)\s*:.*?(?=(?:coverage|benefit|exclusion|condition|term)|$)',
            r'(?i)(?:whereas|provided that|subject to|in the event|notwithstanding).*?(?=\.|;)',
            r'\d+\.\s+.*?(?=\d+\.|$)',  # Numbered clauses
            r'[A-Z][^.!?]*(?:shall|will|must|may|should)[^.!?]*[.!?]',  # Legal language
        ]
        
        for pattern in clause_patterns:
            matches = re.finditer(pattern, text, re.MULTILINE | re.DOTALL)
            for match in matches:
                clause = match.group().strip()
                if len(clause) > 50:  # Filter out very short matches
                    clauses.append(clause)
        
        return clauses
    
    def chunk_by_sentences(self, text: str, max_tokens: int = 150, overlap: int = 25) -> List[Dict[str, str]]:
        """Chunk text by sentences with reduced size for token optimization"""
        self._download_nltk_data()
        
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for i, sentence in enumerate(sentences):
            # More accurate token count estimation
            sentence_tokens = len(sentence.split()) * 1.2
            
            if current_length + sentence_tokens > max_tokens and current_chunk:
                # Create focused chunk
                chunk_text = ' '.join(current_chunk).strip()
                # Only create chunk if it has substantial content
                if len(chunk_text) > 100:  # Minimum chunk size
                    chunks.append({
                        'text': chunk_text,
                        'type': 'optimized_chunk',
                        'token_count': int(current_length),
                        'sentence_range': f"{i - len(current_chunk)}-{i - 1}"
                    })
                
                # Reduced overlap to save tokens
                overlap_count = min(2, len(current_chunk) // 3)
                overlap_sentences = current_chunk[-overlap_count:] if overlap_count > 0 else []
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s.split()) * 1.2 for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_tokens
        
        # Add remaining chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk).strip()
            if len(chunk_text) > 100:
                chunks.append({
                    'text': chunk_text,
                    'type': 'optimized_chunk',
                    'token_count': int(current_length),
                    'sentence_range': 'final'
                })
        
        return chunks
    
    def chunk_by_clauses(self, text: str) -> List[Dict[str, str]]:
        """Chunk text by detected clauses"""
        clauses = self.detect_clauses(text)
        chunks = []
        
        for i, clause in enumerate(clauses):
            chunks.append({
                'text': clause,
                'type': 'clause',
                'clause_id': i
            })
        
        return chunks
    
    def semantic_chunking(self, text: str) -> List[Dict[str, str]]:
        """Advanced semantic chunking using spaCy"""
        if not self.nlp:
            # Fallback to sentence chunking
            return self.chunk_by_sentences(text)
        
        doc = self.nlp(text)
        chunks = []
        current_chunk = []
        current_topic = None
        
        for sent in doc.sents:
            # Simple topic detection based on key entities and keywords
            entities = [ent.label_ for ent in sent.ents]
            keywords = [token.lemma_.lower() for token in sent if token.pos_ in ['NOUN', 'PROPN']]
            
            sent_topic = self._get_topic_from_keywords(keywords + entities)
            
            if current_topic is None:
                current_topic = sent_topic
                current_chunk.append(sent.text)
            elif sent_topic == current_topic or len(current_chunk) < 3:
                current_chunk.append(sent.text)
            else:
                # Topic change - create chunk
                if current_chunk:
                    chunks.append({
                        'text': ' '.join(current_chunk),
                        'type': 'semantic_chunk',
                        'topic': current_topic
                    })
                
                current_chunk = [sent.text]
                current_topic = sent_topic
        
        # Add remaining chunk
        if current_chunk:
            chunks.append({
                'text': ' '.join(current_chunk),
                'type': 'semantic_chunk',
                'topic': current_topic
            })
        
        return chunks
    
    def _get_topic_from_keywords(self, keywords: List[str]) -> str:
        """Determine topic from keywords"""
        # Define topic keywords for insurance/legal documents
        topic_mapping = {
            'coverage': ['coverage', 'benefit', 'claim', 'policy', 'insured'],
            'exclusion': ['exclusion', 'excluded', 'not covered', 'limitation'],
            'premium': ['premium', 'payment', 'due', 'billing', 'fee'],
            'medical': ['medical', 'health', 'doctor', 'hospital', 'treatment'],
            'legal': ['legal', 'court', 'lawsuit', 'liability', 'responsibility'],
            'general': []
        }
        
        for topic, topic_keywords in topic_mapping.items():
            if any(keyword in keywords for keyword in topic_keywords):
                return topic
        
        return 'general'
    
    def create_chunks(self, text: str, chunk_size: int = 200, overlap: int = 50) -> List[Dict[str, str]]:
        """Main method to create intelligent chunks"""
        # Preprocess text
        text = self.preprocess_text(text)
        
        # Combine different chunking strategies
        all_chunks = []
        
        # 1. Sentence-based chunks
        sentence_chunks = self.chunk_by_sentences(text, chunk_size, overlap)
        all_chunks.extend(sentence_chunks)
        
        # 2. Clause-based chunks
        clause_chunks = self.chunk_by_clauses(text)
        all_chunks.extend(clause_chunks)
        
        # 3. Semantic chunks (if spaCy is available)
        if self.nlp:
            semantic_chunks = self.semantic_chunking(text)
            all_chunks.extend(semantic_chunks)
        
        # Remove duplicates and very short chunks
        unique_chunks = []
        seen_texts = set()
        
        for chunk in all_chunks:
            chunk_text = chunk['text'].strip()
            if len(chunk_text) > 30 and chunk_text not in seen_texts:
                seen_texts.add(chunk_text)
                unique_chunks.append(chunk)
        
        return unique_chunks

# Convenience function
def create_chunks(text: str, chunk_size: int = 200, overlap: int = 50) -> List[Dict[str, str]]:
    """Create intelligent text chunks"""
    chunker = TextChunker()
    return chunker.create_chunks(text, chunk_size, overlap)
