# ELA Journal Vector Search and RAG Evaluation Framework

## 📝 Deliverables for Grader

### [🎥 Watch the Project Demo Video](https://www.loom.com/share/e59a360d01f740a18fc7ecdda81f1c6f)
A 5-minute walkthrough of the system in action, demonstrating the RAG pipeline and evaluation framework.

### [📚 Read the Homework Questions & Answers](homework_questions.md)
Comprehensive documentation of the project's goals, implementation, and results.

---

This project creates a searchable vector database from ELA (Estudios de Lingüística Aplicada) journal PDFs using Qdrant and LangChain. It includes a comprehensive RAG (Retrieval-Augmented Generation) evaluation framework that uses RAGAS to assess the quality of the system's responses.

## Project Overview

The project consists of several key components that work together to create, evaluate, and optimize a RAG system:

1. **Golden Test Generation** (`01_chunk_generate_embeddings_and_golden_test_set`)
   - Uses RAGAS to create a test dataset
   - Models used:
     - Embeddings: OpenAI Embeddings (text-embedding-3-small)
     - Test Generation: GPT-3.5 Turbo (gpt-3.5-turbo-0125)
   - Processes PDF documents and generates embeddings
   - Saves embeddings and metadata in `embeddings_cache` for reuse
   - Creates a CSV file with fields: user_input (the golden query), reference_contexts (the golden contexts) and reference (the golden answer)
   - Concept: the golden test set is the ground truth. This set provides a reliable benchmark for evaluating the RAG system's retrieval and generation capabilities. By comparing system outputs to the golden set, you can objectively measure improvements and identify areas for further optimization.

2. **Vector Database Population** (`02_vector_database_population.py`)
   - Loads the cached embeddings from the previous step
   - Uses Qdrant as the vector database (with cosine similarity)
   - Vector size: 1536 dimensions (compatible with OpenAI embeddings)
   - Handles batch processing and concurrent uploads for efficiency
   - Ensures proper text cleaning and metadata preservation

3. **Prepare a Data Set to Evaluate a Speicifc RAG Iteration** (`generate_golden_test_results.py`)
   - Runs the RAG system against the golden test dataset
   - Uses the inference chain defined in `rag_graph.py`
   - Generates responses for each test case using the rag_graph: retrieved_contexts (input chunks) and response (outut from inference) are appended to a new CSV
   - Saves results in a CSV file for evaluation

4. **RAG Evaluation** (`evaluate_rag.py`)
   - Uses RAGAS to evaluate the RAG system's performance
   - Models used:
     - Evaluation LLM: GPT-3.5 Turbo (gpt-3.5-turbo-0125)
   - Implements multiple evaluation metrics:
     - Context Recall
     - Faithfulness
     - Factual Correctness
     - Response Relevancy
     - Entity Recall
     - Noise Sensitivity
   - Provides detailed scoring and analysis
   - Uses a 360-second timeout for comprehensive evaluation

The core RAG implementation (`rag_graph.py`) handles the actual inference process, including context retrieval, prompt construction, and response generation.

#### 

Prerequisites

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
uv sync
```

3. **Clearing Qdrant**

If you need to clear the vector database and start fresh:
```bash
python clear_qdrant.py
```


# Issues that I still have not worked through

I was getting the below exceptios when running the evaluations:s
- ████████████████████████████▉               | 221/300 [03:02<01:07,  1.17it/s]Exception raised in Job[197]: ValueError(setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (3,) + inhomogeneous part.)


Exception raised in Job[220]: LLMDidNotFinishException(The LLM generation was not completed. Please increase try increasing the max_tokens and try again.)


https://github.com/urllib3/urllib3/issues/3020