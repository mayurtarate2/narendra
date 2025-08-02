from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class DocumentInput(BaseModel):
    documents: str = Field(..., description="URL to a PDF, DOCX, or email content")
    questions: List[str] = Field(..., description="Array of natural language questions")

class QueryResponse(BaseModel):
    answers: List[str] = Field(..., description="Array of answers corresponding to each question")

class DocumentLog(BaseModel):
    id: int
    document_url: str
    upload_timestamp: datetime
    status: str

class QuestionLog(BaseModel):
    id: int
    document_id: int
    question: str
    answer: str
    timestamp: datetime
