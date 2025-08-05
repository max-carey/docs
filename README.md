# ELA Journal Vector Search

This project creates a searchable vector database from ELA (Estudios de Lingüística Aplicada) journal PDFs using Qdrant and LangChain.

## Prerequisites

- Docker Desktop for Mac
- Python 3.9+
- uv (for dependency management)

## Setup

1. **Start Qdrant**

```bash
# Pull the latest Qdrant image
docker pull qdrant/qdrant

# Run Qdrant container
# Option 1: Run in the foreground (stop with Ctrl+C)
docker run -p 6333:6333 -p 6334:6334 \
    -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
    qdrant/qdrant

# Option 2: Run in the background
docker run -d -p 6333:6333 -p 6334:6334 \
    -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
    qdrant/qdrant
```

Qdrant will be accessible at:
- REST API: http://localhost:6333
- Web UI: http://localhost:6333/dashboard
- GRPC API: http://localhost:6334

2. **Install Python Dependencies**

```bash
# Install dependencies using uv
uv pip sync
```

## Usage

1. **Download PDFs**
```bash
python scrape_ela.py
```

2. **Process PDFs and Create Vector Database**
```bash
python process_pdfs.py
```

## Project Structure

- `scrape_ela.py` - Downloads PDFs from ELA journal
- `process_pdfs.py` - Processes PDFs and creates vector embeddings
- `qdrant_storage/` - Local storage for Qdrant vector database
- `downloads/ela_issues/` - Downloaded PDF files

## Security Note

The default Qdrant setup has no authentication. For production use, refer to [Qdrant's security documentation](https://qdrant.tech/documentation/guides/security).