from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.environ["PINECONE_API_KEY"]

import os
from pinecone import Pinecone, ServerlessSpec
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

if "ragtest" not in pc.list_indexes().names():
    pc.create_index(
        name="rag-test",
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

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
print(index.describe_index_stats())

