# Document Q&A

A document question-answering application built with Streamlit, FAISS, and Groq.
Upload any PDF directly from the browser and ask questions about it using a
retrieval augmented generation pipeline.

## Features

- Upload any PDF directly from the browser
- Automatic chunking and vector indexing on upload
- Sidebar showing loaded document name and total chunks indexed
- Conversational chat interface with message history
- Clear chat button to reset the conversation

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

Create a .env file in the root directory:

    GROQ_API_KEY=your-groq-api-key-here

Run the application:

    streamlit run app.py

Upload a PDF using the sidebar and start asking questions.

## Project Structure

    app.py              Streamlit UI, PDF processing, and chat interface
    retriever.py        Semantic search over the FAISS index
    requirements.txt    Python dependencies
    faiss_index/        Auto-generated vector store (created on first upload)

## Versions

- v1.0.0 Initial working version with manual PDF ingestion
- v1.1.0 Added browser PDF uploader, sidebar with document info, and clear chat button