from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.environ["PINECONE_API_KEY"]

import os
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

if "ragtest" not in pc.list_indexes().names():
    pc.create_index(
        name="rag-test",
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index("ragtest")
print(index.describe_index_stats())