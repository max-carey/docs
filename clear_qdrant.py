import logging
from qdrant_client import QdrantClient
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load environment variables
load_dotenv()

# Constants
COLLECTION_NAME = "linguistics_articles"

def clear_qdrant():
    """Delete all vectors from Qdrant collection."""
    try:
        # Initialize Qdrant client
        client = QdrantClient(url="http://localhost:6333")
        
        # Check if collection exists
        collections = client.get_collections().collections
        exists = any(col.name == COLLECTION_NAME for col in collections)
        
        if exists:
            # Delete the collection
            client.delete_collection(collection_name=COLLECTION_NAME)
            logging.info(f"Successfully deleted collection: {COLLECTION_NAME}")
        else:
            logging.info(f"Collection {COLLECTION_NAME} does not exist")
            
    except Exception as e:
        logging.error(f"Error clearing Qdrant: {str(e)}")

if __name__ == "__main__":
    clear_qdrant()