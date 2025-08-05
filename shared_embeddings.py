import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

# Constants
EMBEDDING_MODEL = "text-embedding-3-small"  # Using smaller, more cost-effective OpenAI embedding model
EMBEDDINGS_CACHE_DIR = Path("embeddings_cache")
EMBEDDINGS_CACHE_DIR.mkdir(exist_ok=True)

def get_embeddings() -> Embeddings:
    """Get shared embeddings instance."""
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        dimensions=1536  # text-embedding-3-small uses 1536 dimensions
    )

def get_chat_model():
    """Get shared chat model instance."""
    return ChatOpenAI(model="gpt-4")

def save_document_embeddings(docs: List[Document], vectors: List[float], cache_key: str):
    """Save document embeddings to cache.
    
    Args:
        docs: List of documents
        vectors: List of embedding vectors
        cache_key: Unique identifier for this set of embeddings
    """
    cache_file = EMBEDDINGS_CACHE_DIR / f"{cache_key}.pkl"
    metadata_file = EMBEDDINGS_CACHE_DIR / f"{cache_key}_metadata.json"
    
    # Save vectors and documents
    with open(cache_file, 'wb') as f:
        pickle.dump({'docs': docs, 'vectors': vectors}, f)
    
    # Save metadata separately for quick checking
    metadata = {
        'num_docs': len(docs),
        'vector_dim': len(vectors[0]) if vectors else 0,
        'doc_ids': [doc.metadata.get('source', '') for doc in docs]
    }
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f)

def load_document_embeddings(cache_key: str) -> Optional[Tuple[List[Document], List[float]]]:
    """Load document embeddings from cache if they exist.
    
    Args:
        cache_key: Unique identifier for the embeddings to load
        
    Returns:
        Tuple of (documents, vectors) if found, None if not found
    """
    cache_file = EMBEDDINGS_CACHE_DIR / f"{cache_key}.pkl"
    
    if not cache_file.exists():
        return None
        
    with open(cache_file, 'rb') as f:
        data = pickle.load(f)
        return data['docs'], data['vectors']