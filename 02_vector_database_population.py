import logging
import asyncio
from typing import Dict, List

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm.asyncio import tqdm_asyncio

from shared_embeddings import load_document_embeddings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

COLLECTION_NAME = "linguistics_articles"
BATCH_SIZE = 100
MAX_CONCURRENT_TASKS = 5

def init_qdrant() -> QdrantClient:
    """Initialize Qdrant client and create collection if it doesn't exist."""
    client = QdrantClient(url="http://localhost:6333")
    
    collections = client.get_collections().collections
    exists = any(col.name == COLLECTION_NAME for col in collections)
    
    if not exists:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=1536,
                distance=Distance.COSINE
            ),
        )
        logging.info(f"Created new collection: {COLLECTION_NAME}")
    
    return client

def clean_text(text: str) -> str:
    """Clean text by removing invalid Unicode characters."""
    return text.encode('utf-8', errors='ignore').decode('utf-8')

def create_embeddings(cache_key: str = "golden_test") -> List[Dict]:
    """Load cached embeddings from golden test set. Raises error if not found."""
    cached = load_document_embeddings(cache_key)
    
    if not cached:
        raise ValueError(f"No cached embeddings found with key '{cache_key}'")
    
    cached_docs, cached_vectors = cached
    logging.info(f"Loaded {len(cached_docs)} cached embeddings")
    
    return [
        {
            "id": i,
            "vector": vector,
            "payload": {
                "text": clean_text(doc.page_content),
                "metadata": doc.metadata
            }
        }
        for i, (doc, vector) in enumerate(zip(cached_docs, cached_vectors))
    ]

def clean_text(text: str) -> str:
    """Clean text by removing invalid Unicode characters."""
    return text.encode('utf-8', errors='ignore').decode('utf-8')

async def upload_batch(client: QdrantClient, batch: List[Dict], semaphore: asyncio.Semaphore) -> None:
    """Upload a batch of vectors to Qdrant."""
    async with semaphore:
        points = [
            PointStruct(
                id=v["id"],
                vector=v["vector"],
                payload={
                    "text": clean_text(v["payload"]["text"]),
                    "metadata": v["payload"]["metadata"]
                }
            )
            for v in batch
        ]
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
                wait=True
            )
        )

async def upload_to_qdrant_async(client: QdrantClient, vectors: List[Dict]):
    """Upload vectors to Qdrant in parallel batches."""
    batches = [vectors[i:i + BATCH_SIZE] for i in range(0, len(vectors), BATCH_SIZE)]
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
    
    await tqdm_asyncio.gather(
        *[upload_batch(client, batch, semaphore) for batch in batches],
        desc=f"Uploading vectors in batches of {BATCH_SIZE}"
    )
    
    logging.info(f"Uploaded {len(vectors)} vectors to Qdrant")

async def main_async():
    """Async main function to load cached embeddings and create vector database."""
    client = init_qdrant()
    
    vectors = create_embeddings()
    await upload_to_qdrant_async(client, vectors)

def main():
    """Main function that runs the async event loop."""
    asyncio.run(main_async())

if __name__ == "__main__":
    main()