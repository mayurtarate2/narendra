# HackRX Intelligent Document Q&A System

A production-ready, end-to-end LLM-powered intelligent document question-answering system built with FastAPI, OpenAI GPT-4, and ChromaDB.

## 🎯 Features

- **Multi-format Document Support**: PDF, DOCX, and email content parsing
- **Intelligent Chunking**: spaCy and NLTK-powered clause-level text processing
- **Semantic Search**: ChromaDB vector storage with OpenAI embeddings
- **GPT-4 Powered Answers**: Accurate, explainable responses with source citations
- **RESTful API**: FastAPI-based with automatic documentation
- **Bearer Token Auth**: Secure API access
- **SQLite Logging**: Document and query logging
- **Fast Performance**: <5s average response time

## 🛠 Tech Stack

- **Backend**: FastAPI + Uvicorn
- **LLM & Embeddings**: OpenAI GPT-4 + text-embedding-3-small
- **Vector Database**: ChromaDB (local, no external API required)
- **Traditional Database**: SQLite
- **Document Processing**: PyMuPDF, python-docx, email.parser
- **NLP**: spaCy, NLTK
- **Data Validation**: Pydantic

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Make setup script executable
chmod +x setup.sh

# Run setup (creates venv, installs dependencies, downloads models)
./setup.sh
```

### 2. Configure Environment

Edit `.env` file:
```bash
OPENAI_API_KEY=your_openai_api_key_here
BEARER_TOKEN=your_secure_bearer_token_here
```

### 3. Run the Server

```bash
# Activate virtual environment
source venv/bin/activate

# Start the server
python main.py
```

The server will start on `http://localhost:8000`

API Documentation: `http://localhost:8000/docs`

## 📡 API Usage

### Main Endpoint: `/hackrx/run`

**Request:**
```json
POST /hackrx/run
Authorization: Bearer your_token_here
Content-Type: application/json

{
  "documents": "https://example.com/sample-policy.pdf",
  "questions": [
    "Does this policy cover knee surgery and under what conditions?",
    "What is the grace period for premium payment?"
  ]
}
```

**Response:**
```json
{
  "success": true,
  "answers": [
    {
      "question": "Does this policy cover knee surgery and under what conditions?",
      "answer": "Yes, knee surgery is covered under the medical benefits section...",
      "clause": "Section 4.2: Surgical procedures including orthopedic surgeries...",
      "rationale": "The policy explicitly lists orthopedic surgeries under covered procedures..."
    }
  ],
  "processing_time": 3.45,
  "document_id": "doc_a1b2c3d4_1234567890"
}
```

### Testing with cURL

```bash
curl -X POST "http://localhost:8000/hackrx/run" \
     -H "Authorization: Bearer your_token_here" \
     -H "Content-Type: application/json" \
     -d '{
       "documents": "https://example.com/sample-policy.pdf",
       "questions": [
         "What is covered under this policy?",
         "What are the exclusions?"
       ]
     }'
```

## 🏗 System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   Document       │    │   Text          │
│   Routes        │───▶│   Parser         │───▶│   Chunker       │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Response      │    │   GPT-4          │    │   ChromaDB      │
│   Generator     │◀───│   LLM Service    │◀───│   Embeddings    │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │
         ▼
┌─────────────────┐
│   SQLite        │
│   Logging       │
│                 │
└─────────────────┘
```

## 📁 Project Structure

```
hackrx-system/
├── main.py              # FastAPI application entry point
├── routes.py            # API endpoints and request handling
├── models.py            # Pydantic data models
├── parser.py            # Document parsing (PDF, DOCX, email)
├── chunker.py           # Intelligent text chunking
├── embedding.py         # OpenAI embeddings + ChromaDB
├── llm.py               # GPT-4 answer generation
├── database.py          # SQLite operations
├── utils.py             # Utility functions
├── requirements.txt     # Python dependencies
├── setup.sh             # Setup script
├── .env.example         # Environment variables template
└── README.md            # This file
```

## 🔧 Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `BEARER_TOKEN`: API authentication token (optional)
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)
- `DEBUG`: Debug mode (default: false)

### Performance Tuning

- **Chunk Size**: Adjust in `chunker.py` (default: 200 tokens)
- **Chunk Overlap**: Adjust overlap for better context (default: 50 tokens)
- **Top-K Results**: Number of relevant chunks to retrieve (default: 5)
- **GPT-4 Temperature**: Control response creativity (default: 0.1 for accuracy)

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/hackrx/run` | POST | Main Q&A endpoint |
| `/hackrx/health` | GET | Health check |
| `/hackrx/stats` | GET | System statistics |
| `/hackrx/parse-only` | POST | Parse document only (testing) |
| `/docs` | GET | API documentation |

## 🔒 Security

- Bearer token authentication
- Input validation with Pydantic
- Error handling and sanitization
- Rate limiting (implement as needed)

## 🚀 Deployment

### Local Development
```bash
python main.py
```

### Production (with Gunicorn)
```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000
```

### Docker
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_sm
COPY . .
CMD ["python", "main.py"]
```

### Railway/Render Deployment
1. Connect your GitHub repository
2. Set environment variables (OPENAI_API_KEY, etc.)
3. Deploy with the provided `main.py`

## 🧪 Testing

### Sample Documents for Testing

1. **Insurance Policy PDF**: Test with real insurance documents
2. **Legal Contract DOCX**: Upload contract documents
3. **Email Content**: Paste email text directly

### Sample Questions

- "What is the coverage limit for medical expenses?"
- "What are the exclusions in this policy?"
- "What is the grace period for premium payments?"
- "Under what conditions can the policy be cancelled?"

### Performance Benchmarks

- Average response time: <5 seconds
- Document parsing: <2 seconds
- Chunk creation: <1 second
- Embedding generation: <2 seconds
- Answer generation: <3 seconds

## 🔍 Troubleshooting

### Common Issues

1. **spaCy Model Missing**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **NLTK Data Missing**:
   ```bash
   python -c "import nltk; nltk.download('punkt')"
   ```

3. **OpenAI API Errors**:
   - Check API key validity
   - Verify account has credits
   - Check rate limits

4. **Document Parsing Fails**:
   - Ensure document URL is accessible
   - Check document format (PDF, DOCX supported)
   - Verify file is not corrupted

### Logs and Monitoring

- Check console output for processing logs
- SQLite database stores all queries and responses
- ChromaDB persists in `./chroma_db` directory

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 💡 Advanced Features

### Custom Document Types
Extend `parser.py` to support additional document formats.

### Enhanced Chunking
Implement domain-specific chunking strategies in `chunker.py`.

### Multiple LLM Support
Add support for other LLMs by extending `llm.py`.

### Caching
Implement Redis caching for frequently asked questions.

### Async Processing
Add background job processing for large documents.

---

**Built for HackRX Challenge** 🏆

For support or questions, please create an issue in the repository.
