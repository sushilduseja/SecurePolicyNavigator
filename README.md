# Secure Policy Navigator

A RAG-based Compliance Assistant for Security Policies & Standards that helps users navigate and understand complex compliance documentation through natural language queries.

## Features

- Natural language querying of security compliance documents
- Local LLM integration via Ollama
- Efficient document retrieval using RAG architecture
- Clean, intuitive Streamlit interface
- Local vector storage with ChromaDB

## Project Structure

```
secure-policy-navigator/
├── data/               # Sample compliance documents
├── src/               # Source code
│   ├── components/    # Reusable components
│   ├── rag/          # RAG pipeline implementation
│   └── utils/        # Utility functions
├── tests/            # Test files
├── .gitignore        # Git ignore file
├── requirements.txt  # Project dependencies
└── README.md        # Project documentation
```

## Setup

1. Ensure Python 3.9+ is installed
2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/Mac:
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Install and start Ollama (follow instructions at https://ollama.ai)

## Running the Application

Run the application using:
```bash
streamlit run src/app.py
```

## Development

- Place compliance documents in the `data/` directory
- Main application code is in `src/app.py`
- RAG pipeline implementation is in `src/rag/`
- Reusable components are in `src/components/`

## License

MIT License