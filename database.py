import sqlite3
import os
from datetime import datetime
from typing import List, Optional
import hashlib
from functools import lru_cache

DATABASE_PATH = "hackrx_system.db"

def init_database():
    """Initialize high-performance SQLite database"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Performance optimizations
    cursor.execute("PRAGMA journal_mode = WAL")  # Write-Ahead Logging
    cursor.execute("PRAGMA synchronous = OFF")   # Faster writes
    cursor.execute("PRAGMA temp_store = MEMORY") # Use memory for temp operations
    cursor.execute("PRAGMA cache_size = 10000")  # Larger cache
    cursor.execute("PRAGMA mmap_size = 268435456") # Memory-mapped I/O (256MB)
    
    # Create documents table with indexes
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_url TEXT NOT NULL,
            document_hash TEXT UNIQUE,
            upload_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'processed'
        )
    """)
    
    # Create index on hash for fast lookups
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_document_hash ON documents(document_hash)
    """)
    
    # Create questions table with indexes
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            question_hash TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (document_id) REFERENCES documents (id)
        )
    """)
    
    conn.commit()
    conn.close()

def log_document(document_url: str) -> int:
    """Log document upload and return document ID"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute(
        "INSERT INTO documents (document_url) VALUES (?)",
        (document_url,)
    )
    
    document_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return document_id

def log_question(document_id: int, question: str, answer: str):
    """Log question and answer"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute(
        "INSERT INTO questions (document_id, question, answer) VALUES (?, ?, ?)",
        (document_id, question, answer)
    )
    
    conn.commit()
    conn.close()

def get_document_logs() -> List[dict]:
    """Get all document logs"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM documents ORDER BY upload_timestamp DESC")
    rows = cursor.fetchall()
    
    conn.close()
    
    return [
        {
            "id": row[0],
            "document_url": row[1],
            "upload_timestamp": row[2],
            "status": row[3]
        }
        for row in rows
    ]

def get_question_logs(document_id: Optional[int] = None) -> List[dict]:
    """Get question logs, optionally filtered by document_id"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    if document_id:
        cursor.execute(
            "SELECT * FROM questions WHERE document_id = ? ORDER BY timestamp DESC",
            (document_id,)
        )
    else:
        cursor.execute("SELECT * FROM questions ORDER BY timestamp DESC")
    
    rows = cursor.fetchall()
    conn.close()
    
    return [
        {
            "id": row[0],
            "document_id": row[1],
            "question": row[2],
            "answer": row[3],
            "timestamp": row[4]
        }
        for row in rows
    ]
