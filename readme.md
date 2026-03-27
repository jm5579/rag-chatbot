# RAG Chatbot

A document question-answering application built with Streamlit, FAISS, and Groq.
Upload any PDF and ask questions about it in natural language using a retrieval
augmented generation pipeline.

## How It Works

The application splits a PDF into text chunks and converts them into vector
embeddings using a sentence transformer model. These embeddings are stored in a
FAISS vector database. When a user submits a question, the most semantically
similar chunks are retrieved and passed to an LLM alongside the question to
generate a grounded, context-aware answer.

## Tech Stack

- Python 3.11
- Streamlit
- LangChain
- FAISS
- HuggingFace sentence-transformers (all-MiniLM-L6-v2)
- Groq API (llama-3.1-8b-instant)

## Setup

Clone the repository:

    git clone https://github.com/your-username/rag-chatbot.git
    cd rag-chatbot

Install dependencies:

    pip install -r requirements.txt

Create a .env file in the root directory with your Groq API key:

    GROQ_API_KEY=your-groq-api-key-here

Place a PDF file inside the data/ directory.

Run the ingestion script to chunk the document and build the vector index:

    python ingest.py

Launch the application:

    streamlit run app.py

## Project Structure

    app.py              Streamlit UI and chat interface
    ingest.py           PDF loading, chunking, and FAISS index creation
    retriever.py        Semantic search over the FAISS index
    requirements.txt    Python dependencies
    data/               Directory for input PDF files
    faiss_index/        Auto-generated vector store (created after running ingest.py)

## Notes

The faiss_index/ and data/ directories are excluded from version control.
Each user must run ingest.py locally after adding their own PDF to the data/ folder.
The embedding model is downloaded automatically on first run.