from openai import OpenAI
import chromadb
from chromadb.config import Settings
import numpy as np
import faiss
import pickle
import os
from typing import List, Dict, Tuple
import uuid
from dotenv import load_dotenv
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib

load_dotenv()

class HighPerformanceEmbeddingService:
    """Ultra-fast embedding service with FAISS and caching"""
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Initialize FAISS index
        self.dimension = 512  # Reduced dimensions for speed
        self.faiss_index = faiss.IndexHNSWFlat(self.dimension, 32)  # HNSW for speed
        self.faiss_index.hnsw.efConstruction = 40
        self.faiss_index.hnsw.efSearch = 16  # Lower for faster search
        
        # Metadata storage
        self.chunk_metadata = {}
        self.document_chunks = {}
        
        # Caches
        self.embedding_cache = {}
        self.index_path = "faiss_index.pkl"
        self.metadata_path = "chunk_metadata.pkl"
        
        # Load existing index if available
        self._load_index()
        
        # Thread pool for async operations
        self.thread_pool = ThreadPoolExecutor(max_workers=2)
    
    def _load_index(self):
        """Load existing FAISS index and metadata"""
        try:
            if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
                with open(self.index_path, 'rb') as f:
                    index_data = pickle.load(f)
                    if index_data['vectors'].shape[0] > 0:
                        self.faiss_index.add(index_data['vectors'])
                
                with open(self.metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                    self.chunk_metadata = metadata.get('chunks', {})
                    self.document_chunks = metadata.get('documents', {})
                
                print(f"Loaded FAISS index with {self.faiss_index.ntotal} vectors")
        except Exception as e:
            print(f"Could not load existing index: {e}")
    
    def _save_index(self):
        """Save FAISS index and metadata"""
        try:
            # Save vectors
            vectors = np.array([self.faiss_index.reconstruct(i) for i in range(self.faiss_index.ntotal)])
            with open(self.index_path, 'wb') as f:
                pickle.dump({'vectors': vectors}, f)
            
            # Save metadata
            with open(self.metadata_path, 'wb') as f:
                pickle.dump({
                    'chunks': self.chunk_metadata,
                    'documents': self.document_chunks
                }, f)
        except Exception as e:
            print(f"Could not save index: {e}")
    
    def get_embedding_hash(self, text: str) -> str:
        """Generate hash for embedding caching"""
        return hashlib.md5(text.encode()).hexdigest()[:12]

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI's text-embedding-3-small with caching"""
        try:
            if not texts:
                return []
            
            # Check cache first
            embeddings = []
            uncached_texts = []
            uncached_indices = []
            
            for i, text in enumerate(texts):
                text_hash = self.get_embedding_hash(text)
                if text_hash in self.embedding_cache:
                    embeddings.append(self.embedding_cache[text_hash])
                else:
                    embeddings.append(None)
                    uncached_texts.append(text[:1500])  # Truncate for speed
                    uncached_indices.append(i)
            
            # Generate embeddings for uncached texts
            if uncached_texts:
                response = self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=uncached_texts,
                    dimensions=512  # Reduced dimensions for speed
                )
                
                # Cache and store results
                for idx, embedding_obj in enumerate(response.data):
                    original_idx = uncached_indices[idx]
                    embedding = embedding_obj.embedding
                    
                    # Cache the embedding
                    text_hash = self.get_embedding_hash(texts[original_idx])
                    self.embedding_cache[text_hash] = embedding
                    
                    # Store in results
                    embeddings[original_idx] = embedding
            
            return embeddings
            
        except Exception as e:
            print(f"Error generating embeddings: {str(e)}")
            # Return zero vectors as fallback
            return [[0.0] * self.dimension for _ in texts]
    
    def store_chunks_faiss(self, chunks: List[Dict], document_id: str) -> List[str]:
        """Store document chunks in FAISS index"""
        try:
            if not chunks:
                return []
            
            # Extract texts and generate embeddings
            texts = [chunk['text'] for chunk in chunks]
            embeddings = self.generate_embeddings(texts)
            
            if not embeddings or not embeddings[0]:
                return []
            
            # Convert to numpy array and add to FAISS
            embedding_matrix = np.array(embeddings, dtype=np.float32)
            start_idx = self.faiss_index.ntotal
            self.faiss_index.add(embedding_matrix)
            
            # Store metadata
            chunk_ids = []
            for i, chunk in enumerate(chunks):
                chunk_id = str(uuid.uuid4())
                chunk_ids.append(chunk_id)
                
                # Store chunk metadata
                self.chunk_metadata[start_idx + i] = {
                    'chunk_id': chunk_id,
                    'document_id': document_id,
                    'text': chunk['text'],
                    'type': chunk.get('type', 'text'),
                    'page': chunk.get('page', 0)
                }
            
            # Track document chunks
            if document_id not in self.document_chunks:
                self.document_chunks[document_id] = []
            self.document_chunks[document_id].extend(chunk_ids)
            
            # Save index periodically
            if self.faiss_index.ntotal % 100 == 0:
                self._save_index()
            
            print(f"Stored {len(chunk_ids)} chunks in FAISS for document {document_id}")
            return chunk_ids
            
        except Exception as e:
            print(f"Error storing chunks in FAISS: {str(e)}")
            return []
    
    def search_faiss(self, query: str, document_id: str = None, top_k: int = 3) -> List[Dict]:
        """Search similar chunks using FAISS"""
        try:
            if self.faiss_index.ntotal == 0:
                return []
            
            # Generate query embedding
            query_embedding = self.generate_embeddings([query])[0]
            if not query_embedding:
                return []
            
            # Search in FAISS
            query_vector = np.array([query_embedding], dtype=np.float32)
            distances, indices = self.faiss_index.search(query_vector, min(top_k * 3, self.faiss_index.ntotal))
            
            # Filter and format results
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx == -1:  # Invalid index
                    continue
                    
                if idx not in self.chunk_metadata:
                    continue
                
                metadata = self.chunk_metadata[idx]
                
                # Filter by document if specified
                if document_id and metadata['document_id'] != document_id:
                    continue
                
                # Convert distance to similarity score (FAISS returns L2 distance)
                similarity_score = 1.0 / (1.0 + distance)
                
                results.append({
                    'text': metadata['text'],
                    'similarity_score': float(similarity_score),
                    'metadata': {
                        'chunk_id': metadata['chunk_id'],
                        'document_id': metadata['document_id'],
                        'chunk_type': metadata.get('type', 'text'),
                        'page': metadata.get('page', 0)
                    }
                })
                
                if len(results) >= top_k:
                    break
            
            return results
            
        except Exception as e:
            print(f"Error searching FAISS: {str(e)}")
            return []
    
    def get_stats(self) -> Dict:
        """Get performance statistics"""
        return {
            'total_chunks': self.faiss_index.ntotal,
            'total_documents': len(self.document_chunks),
            'cache_size': len(self.embedding_cache),
            'index_type': 'FAISS HNSW',
            'dimensions': self.dimension
        }

# Legacy ChromaDB embedding service (kept for compatibility)
class EmbeddingService:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.chroma_client = chromadb.Client(Settings(allow_reset=True))
        self.collection_name = "document_chunks"
        self._init_collection()
    
    def _init_collection(self):
        """Initialize or get ChromaDB collection"""
        try:
            self.collection = self.chroma_client.get_collection(self.collection_name)
        except:
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI's text-embedding-3-small with optimization"""
        try:
            # Handle empty input
            if not texts:
                return []
            
            # Truncate texts to save tokens (embedding model has 8191 token limit)
            max_chars_per_text = 2000  # ~500 tokens
            truncated_texts = [text[:max_chars_per_text] for text in texts]
            
            # Process in larger batches for efficiency
            batch_size = 200  # Increased batch size for better efficiency
            all_embeddings = []
            
            for i in range(0, len(truncated_texts), batch_size):
                batch = truncated_texts[i:i + batch_size]
                
                response = self.client.embeddings.create(
                    model="text-embedding-3-small",  # Most cost-effective embedding model
                    input=batch,
                    dimensions=512  # Reduce dimensions for cost savings (default is 1536)
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
            
            return all_embeddings
            
        except Exception as e:
            raise Exception(f"Failed to generate embeddings: {str(e)}")
    
    def store_chunks(self, chunks: List[Dict[str, str]], document_id: str) -> List[str]:
        """Store text chunks with embeddings in ChromaDB"""
        try:
            # Prepare data for storage
            texts = [chunk['text'] for chunk in chunks]
            ids = [str(uuid.uuid4()) for _ in chunks]
            
            # Generate embeddings
            embeddings = self.generate_embeddings(texts)
            
            # Prepare metadata
            metadatas = []
            for i, chunk in enumerate(chunks):
                metadata = {
                    'document_id': document_id,
                    'chunk_type': chunk.get('type', 'unknown'),
                    'chunk_index': i
                }
                
                # Add additional metadata based on chunk type
                if chunk.get('topic'):
                    metadata['topic'] = chunk['topic']
                if chunk.get('clause_id') is not None:
                    metadata['clause_id'] = str(chunk['clause_id'])
                if chunk.get('start_sentence') is not None:
                    metadata['start_sentence'] = str(chunk['start_sentence'])
                    metadata['end_sentence'] = str(chunk['end_sentence'])
                
                metadatas.append(metadata)
            
            # Store in ChromaDB
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            return ids
            
        except Exception as e:
            raise Exception(f"Failed to store chunks: {str(e)}")
    
    def search_similar_chunks(self, query: str, document_id: str = None, top_k: int = 5) -> List[Dict]:
        """Search for similar chunks using semantic similarity"""
        try:
            # Generate query embedding
            query_embedding = self.generate_embeddings([query])[0]
            
            # Prepare where clause for filtering by document_id
            where_clause = None
            if document_id:
                where_clause = {"document_id": document_id}
                
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'text': results['documents'][0][i],
                    'similarity_score': 1.0 - results['distances'][0][i],  # Convert distance to similarity
                    'metadata': results['metadatas'][0][i]
                })
            
            return formatted_results
            
        except Exception as e:
            raise Exception(f"Failed to search chunks: {str(e)}")
    
    def get_collection_stats(self):
        """Get collection statistics"""
        try:
            count = self.collection.count()
            return {"total_chunks": count}
        except:
            return {"total_chunks": 0}

# Global high-performance service instance
_high_perf_service = None

def get_embedding_service():
    """Get high-performance embedding service instance"""
    global _high_perf_service
    if _high_perf_service is None:
        _high_perf_service = HighPerformanceEmbeddingService()
    return _high_perf_service

def store_document_chunks(chunks: List[Dict], document_id: str) -> List[str]:
    """Store document chunks using high-performance FAISS service"""
    service = get_embedding_service()
    return service.store_chunks_faiss(chunks, document_id)

def search_relevant_chunks(query: str, document_id: str = None, top_k: int = 3) -> List[Dict]:
    """Search for relevant chunks using high-performance FAISS service"""
    service = get_embedding_service()
    return service.search_faiss(query, document_id, top_k)
