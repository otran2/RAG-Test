# RAG-Test
A local RAG app that connects with pinecone to retrieve stored documents and chunks to generate questions for a user query. An end to end pipeline that handles parsing, chunking, embedding, and storing vectors alongside retrieval, reranking, and generation. 

The data indexing section supports both local file upload and scraping of plain text files. Documents are fed into the system alongside a specified document id and then chunked with a character level tokenizer with the specified chunk size and optional overlap. Chunks are then turned into embeddings using Ollama's nomic-embed-text model, which are then stored into a pinecone vector store. (Note that the user must specify their api key in their own secret .env file to their pinecone database, as private API keys are not provided alogside this project)

The generation is done by retrieving the k (default 15) most relevant chunks from the vector store via cosine similarity, and results are reranked using the ms-marco-MiniLM-L-6-v2 crossencoder. The top k (default 2) are then fed into Ollama's tinyllama model to generate a response.

Note: Before the user can run the program, they must first ensure they download all the requirements listed in requirements.txt, alongside the Ollama web application and its specified models. This can be done using the following steps.
    '''pip install -r requirements.txt
    irm https://ollama.com/install.ps1 | iex
    ollama pull nomic-embed-text
    ollama pull tinyllama'''

To run the application, the user will then run the command below and visit the local port.
    '''uvicorn main:app --reload'''