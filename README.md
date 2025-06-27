# RAG-Document-Q-A-with-GROQ-API-and-Ollama
This Streamlit app uses LangChain, Groq (Gemma2-9b-It), and Ollama (nomic-embed-text) to build a fast and private document Q&amp;A chatbot. Upload your PDFs, embed them using FAISS, and get real-time, accurate answers directly from the documents.

# RAG-Document-Q-A-with-GROQ-API-and-Ollama

This is a Retrieval-Augmented Generation (RAG) PDF Question Answering chatbot built using:

- LangChain for retrieval and document chaining
- GROQ API (Gemma2-9b-It model) for high-performance LLM inference
- Ollama with the `nomic-embed-text` model for local text embeddings
- FAISS for efficient vector storage and retrieval
- Streamlit for a minimal and interactive user interface

---

## Features

- Automatically loads and embeds a folder of PDF documents (`rs_paper/`)
- Performs document chunking and stores them as vectors
- Enables question answering using RAG (retrieval + generation)
- Displays answer and context passage from relevant documents
- Powered by a combination of local and remote components for performance and privacy

---


## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/saurabhhhhhhh/RAG-Document-Q-A-with-GROQ-API-and-Ollama.git
cd RAG-Document-Q-A-with-GROQ-API-and-Ollama
```
### 2. Install dependencies

`pip install -r requirements.txt`

### 3. Create a .env file in the root directory with the following
- GROQ_API_KEY=your_groq_api_key

### 4. Pull the embedding model using Ollama
`ollama pull nomic-embed-text`

### 5. . Add PDF documents
- Create a folder named rs_paper/ and place your PDF files inside it.

### 6. Running the Application
`streamlit run app.py`
