# RAG-For-You
## Application overview
A local RAG app that connects with pinecone to retrieve stored documents and chunks to generate questions for a user query. An end to end pipeline that handles parsing, chunking, embedding, and storing vectors alongside retrieval, reranking, and generation. 

The data indexing section supports both local file upload and scraping of plain text files. Documents are fed into the system alongside a specified document id and then chunked with a character level tokenizer with the specified chunk size and optional overlap. Chunks are then turned into embeddings using Ollama's nomic-embed-text model, which are then stored into a pinecone vector store. (Note that the user must specify their api key in their own secret .env file to their pinecone database, as private API keys are not provided alogside this project)

The generation is done by retrieving the k (default 15) most relevant chunks from the vector store via cosine similarity, and results are reranked using the ms-marco-MiniLM-L-6-v2 crossencoder. The top k (default 2) are then fed into Ollama's tinyllama model to generate a response.

## Getting Started

### Prerequisites & Installation

Before running the program, ensure you download all the requirements listed in `requirements.txt`, alongside the Ollama web application and its specified models. Follow these steps to set up your environment:

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install Ollama (Windows PowerShell):**
   ```powershell
   irm https://ollama.com/install.ps1 | iex
   ```

3. **Download the required local models:**
   ```bash
   ollama pull nomic-embed-text
   ollama pull tinyllama
   ```

---

### Running the Application

Once the setup is complete, initialize the local server by running the command below and visiting the provided local port:

```bash
uvicorn main:app --reload
```
