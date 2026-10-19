from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.environ["PINECONE_API_KEY"]

import os
from pinecone import Pinecone, ServerlessSpec
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import CrossEncoder
import ollama

app = FastAPI()
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

if "ragtest" not in pc.list_indexes().names():
    pc.create_index(
        name="rag-test",
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

class IngestPayload(BaseModel):
    document_id: str
    raw_text: str

class QueryPayload(BaseModel):
    question: str

# Character-based chunker from scratch, taking in chunk size/overlap
def get_chunks(text: str, chunk_size: int = 1000, overlap: int = 200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
index = pc.Index("ragtest")

@app.post("/ingest", summary="Steps 1-3: Load, Chunk, Embed, and Index Data")
def pipeline_ingest(data: IngestPayload):
    # Chunk document
    chunks = get_chunks(data.raw_text)
    if not chunks:
        raise HTTPException(status_code=400, detail="Provided document text is too short to process.")
        
    upsert_batch = []
    
    # Step 3: Loop through, embed locally via Ollama, and stage for indexing
    for idx, chunk in enumerate(chunks):
        try:
            embed_res = ollama.embeddings(model="nomic-embed-text", prompt=chunk)
            vector = embed_res["embedding"]
            
            upsert_batch.append({
                "id": f"{data.document_id}_c{idx}",
                "values": vector,
                "metadata": {
                    "source": data.document_id,
                    "text": chunk
                }
            })
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ollama local embedding crashed: {str(e)}")
            
    # Send vectors up to Pinecone index
    index.upsert(vectors=upsert_batch)
    return {"status": "Success", "chunks_indexed": len(upsert_batch)}


@app.post("/ask", summary="Steps 4-6: Retrieve, Rerank, and Generate Response")
def pipeline_infer(data: QueryPayload):
    # --- STEP 4: SEMANTIC RETRIEVAL ---
    # Convert query text into search coordinates
    query_vector = ollama.embeddings(model="nomic-embed-text", prompt=data.question)["embedding"]
    
    # Retrieve top 15 candidates from Pinecone cloud
    raw_matches = index.query(vector=query_vector, top_k=15, include_metadata=True)
    candidates = raw_matches.get("matches", [])
    
    if not candidates:
        raise HTTPException(status_code=404, detail="No matching documentation stored in cloud database.")
        
    # Extract raw string snippets from retrieval data pool
    retrieved_texts = [match["metadata"]["text"] for match in candidates if "text" in match.get("metadata", {})]
    
    # --- STEP 5: CROSS-ENCODER RERANKING ---
    # Pair up the user question with every single retrieved document candidate
    pairs = [[data.question, text] for text in retrieved_texts]
    
    # Predict high-resolution similarity relevance scores (higher = more relevant)
    rerank_scores = reranker.predict(pairs)
    
    # Zip documents with scores and sort them from highest score to lowest
    scored_docs = sorted(zip(retrieved_texts, rerank_scores), key=lambda x: x[1], reverse=True)
    
    # Cherry-pick only the top 2 absolute best contexts to prevent noise and dilution
    top_reranked_context = [doc[0] for doc in scored_docs[:2]]
    context_block = "\n---\n".join(top_reranked_context)
    
    # --- STEP 6: GROUNDED GENERATION ---
    prompt = f"""You are a precise internal assistant. Review the retrieved data context and answer the user question.
                Stick purely to the facts below. If the answer cannot be confidently answered from the context, respond with:
                "I cannot verify that answer with the existing documentation."

                Retrieved Context:
                {context_block}

                User Question: {data.question}
                Final Synthesized Answer:"""

    try:
        # Ask local LLM to resolve the final prompt construction
        generation_res = ollama.generate(model="tinyllama", prompt=prompt)
        return {
            "answer": generation_res["response"],
            "reranked_context_used": top_reranked_context,
            "raw_retrieval_count": len(retrieved_texts)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama generation down: {str(e)}")

