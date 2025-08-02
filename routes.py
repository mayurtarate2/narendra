from fastapi import APIRouter, HTTPException, Depends, Header, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import List, Optional
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from models import DocumentInput, QueryResponse
from parser import DocumentParser, parsed_cache
from chunker import create_chunks
from embedding import store_document_chunks, search_relevant_chunks, EmbeddingService
from llm import answer_question, LLMService
from database import log_document, log_question, get_document_logs, get_question_logs
from utils import generate_document_id, format_processing_time, safe_json_response
import os
from dotenv import load_dotenv
from functools import lru_cache

load_dotenv()

router = APIRouter()
security = HTTPBearer()

# Thread pool for CPU-intensive tasks
thread_pool = ThreadPoolExecutor(max_workers=4)

# Cache for repeated queries
@lru_cache(maxsize=1000)
def cached_document_id(document_content: str) -> str:
    return generate_document_id(document_content)

# Bearer token validation
def verify_bearer_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify Bearer token authorization"""
    expected_token = os.getenv("BEARER_TOKEN")
    
    if not expected_token:
        # If no token is configured, skip validation
        return True
    
    if credentials.credentials != expected_token:
        raise HTTPException(
            status_code=401, 
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return True

@router.post("/hackrx/run", response_model=QueryResponse)
async def run_query(
    request: DocumentInput,
    background_tasks: BackgroundTasks,
    authorized: bool = Depends(verify_bearer_token)
):
    """
    High-performance async endpoint for intelligent document Q&A
    
    Optimizations:
    - Async I/O for document downloads
    - Multithreading for CPU-intensive parsing
    - Caching for repeated requests
    - Batch processing for multiple questions
    """
    start_time = time.time()
    
    try:
        # Step 1: Check cache first
        content_hash = DocumentParser.get_content_hash(request.documents)
        if content_hash in parsed_cache:
            document_text = parsed_cache[content_hash]
            print(f"Cache hit for document: {content_hash}")
        else:
            # Step 2: Async document parsing
            print(f"Parsing document async: {request.documents[:100]}...")
            
            async with DocumentParser() as parser:
                if request.documents.startswith(('http://', 'https://')):
                    # Async download
                    file_path = await parser.download_file_async(request.documents)
                    # Parse in thread pool to avoid blocking
                    document_text = await asyncio.get_event_loop().run_in_executor(
                        thread_pool, parser.parse_pdf_sync, file_path
                    )
                else:
                    # Direct text processing
                    document_text = request.documents
                
                # Cache the result
                parsed_cache[content_hash] = document_text
        
        if not document_text or len(document_text.strip()) < 50:
            raise HTTPException(
                status_code=400, 
                detail="Document parsing failed or document is too short"
            )
        
        # Step 3: Generate document ID (cached)
        document_id = cached_document_id(request.documents)
        
        # Step 4: Async chunking in thread pool
        print("Creating optimized chunks async...")
        chunks = await asyncio.get_event_loop().run_in_executor(
            thread_pool, create_chunks, document_text, 150, 25
        )
        
        if not chunks:
            raise HTTPException(
                status_code=500,
                detail="Failed to create text chunks from document"
            )
        
        print(f"Created {len(chunks)} chunks")
        
        # Step 5: Async embedding storage
        print("Storing embeddings async...")
        chunk_ids = await asyncio.get_event_loop().run_in_executor(
            thread_pool, store_document_chunks, chunks, document_id
        )
        
        print(f"Stored {len(chunk_ids)} chunks with embeddings")
        
        # Step 6: Process questions in parallel
        answers = []
        llm_service = LLMService()
        
        async def process_question(question: str) -> str:
            """Process individual question asynchronously"""
            try:
                # Preprocess question
                optimized_question = llm_service.preprocess_question(question)
                
                # Search for relevant chunks (async)
                relevant_chunks = await asyncio.get_event_loop().run_in_executor(
                    thread_pool, search_relevant_chunks, optimized_question, document_id, 3
                )
                
                if not relevant_chunks:
                    # Fallback to first chunks
                    relevant_chunks = [
                        {
                            'text': chunk['text'],
                            'metadata': {'chunk_type': chunk.get('type', 'text')},
                            'similarity_score': 0.5
                        }
                        for chunk in chunks[:2]  # Even fewer for speed
                    ]
                
                # Generate answer (async in thread pool)
                llm_response = await asyncio.get_event_loop().run_in_executor(
                    thread_pool, answer_question, optimized_question, relevant_chunks
                )
                
                answer_text = llm_response.get('answer', 'No answer generated')
                
                # Log in background
                background_tasks.add_task(log_question, 1, question, answer_text)
                
                return answer_text
                
            except Exception as e:
                print(f"Error processing question '{question}': {str(e)}")
                return f"Error: {str(e)[:100]}"  # Truncated error
        
        # Process all questions concurrently
        print(f"Processing {len(request.questions)} questions concurrently...")
        answers = await asyncio.gather(*[
            process_question(q) for q in request.questions
        ])
        
        # Step 7: Background logging
        background_tasks.add_task(log_document, request.documents)
        
        # Step 8: Create response
        response = QueryResponse(answers=answers)
        
        processing_time = format_processing_time(start_time)
        print(f"Async request completed in {processing_time}s")
        
        return response
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        
        # Return error response
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/hackrx/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "HackRX Document Q&A System",
        "timestamp": time.time()
    }

@router.get("/hackrx/stats")
async def get_system_stats(authorized: bool = Depends(verify_bearer_token)):
    """Get comprehensive system statistics including token usage"""
    try:
        # Get embedding stats
        embedding_service = EmbeddingService()
        collection_stats = embedding_service.get_collection_stats()
        
        # Get database stats
        document_logs = get_document_logs()
        question_logs = get_question_logs()
        
        # Get token usage stats
        llm_service = LLMService()
        token_stats = llm_service.get_token_usage_stats()
        
        return {
            "documents_processed": len(document_logs),
            "questions_answered": len(question_logs),
            "chunks_stored": collection_stats.get('total_chunks', 0),
            "token_usage": token_stats,
            "optimization_metrics": {
                "avg_chunks_per_query": collection_stats.get('total_chunks', 0) / max(1, len(question_logs)),
                "model_used": "gpt-3.5-turbo (optimized)",
                "embedding_model": "text-embedding-3-small (512 dimensions)"
            },
            "recent_documents": document_logs[:5],
            "recent_questions": question_logs[:10]
        }
    
    except Exception as e:
        return {"error": f"Failed to get stats: {str(e)}"}

@router.post("/hackrx/optimize")
async def optimization_controls(
    action: str,
    authorized: bool = Depends(verify_bearer_token)
):
    """Control optimization settings"""
    llm_service = LLMService()
    
    if action == "clear_cache":
        llm_service.clear_cache()
        return {"message": "LLM cache cleared successfully"}
    elif action == "get_cache_size":
        return {"cache_size": len(llm_service.answer_cache)}
    else:
        return {"error": "Invalid action. Use 'clear_cache' or 'get_cache_size'"}

@router.get("/hackrx/token-usage")
async def get_token_usage(authorized: bool = Depends(verify_bearer_token)):
    """Get detailed token usage statistics"""
    llm_service = LLMService()
    stats = llm_service.get_token_usage_stats()
    
    # Calculate estimated costs (approximate pricing)
    gpt35_cost_per_1k_input = 0.0005  # $0.0005 per 1K input tokens
    gpt35_cost_per_1k_output = 0.0015  # $0.0015 per 1K output tokens
    embedding_cost_per_1k = 0.00002   # $0.00002 per 1K tokens
    
    estimated_llm_cost = (
        (stats['total_prompt_tokens'] / 1000 * gpt35_cost_per_1k_input) +
        (stats['total_completion_tokens'] / 1000 * gpt35_cost_per_1k_output)
    )
    
    return {
        **stats,
        "estimated_costs": {
            "llm_cost_usd": round(estimated_llm_cost, 4),
            "note": "Estimates based on GPT-3.5-turbo pricing"
        },
        "optimization_impact": {
            "model_switch": "GPT-4 → GPT-3.5-turbo (~10x cost reduction)",
            "token_limits": "Context: 1500 tokens, Response: 35 tokens",
            "cache_enabled": True,
            "chunk_optimization": "150 tokens max per chunk"
        }
    }

# Additional utility endpoints

@router.post("/hackrx/parse-only")
async def parse_document_only(
    request: DocumentInput,
    authorized: bool = Depends(verify_bearer_token)
):
    """Parse document without running Q&A (for testing)"""
    try:
        document_text = parse_document(request.documents)
        
        return {
            "success": True,
            "document_length": len(document_text),
            "preview": document_text[:500] + "..." if len(document_text) > 500 else document_text,
            "document_type": "detected based on content analysis"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document parsing failed: {str(e)}")

@router.delete("/hackrx/cleanup")
async def cleanup_old_data(
    authorized: bool = Depends(verify_bearer_token),
    days_old: int = 7
):
    """Clean up old document data (for maintenance)"""
    # This would implement cleanup logic for old documents
    # For now, just return a placeholder response
    return {
        "message": f"Cleanup initiated for data older than {days_old} days",
        "status": "not implemented in this version"
    }
