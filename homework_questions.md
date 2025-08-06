# Homework Questions and Answers

## Task 1: Defining your Problem and Audience

### Problem Description (2 points)
Academic researchers struggle to efficiently search and extract relevant information from Spanish-language linguistics journals due to lack of modern search capabilities. This is especially true in Mexico.

### Why This is a Problem (8 points)
The ELA (Estudios de Lingüística Aplicada) journal contains valuable research in applied linguistics, but its PDF format makes it difficult for researchers to quickly find relevant information. Traditional keyword searches are inadequate for understanding the semantic meaning of research content, and the Spanish language context adds complexity to the search process. Researchers waste significant time manually scanning through PDFs instead of focusing on their research. This particularly impacts linguistics scholars who need to cross-reference methodologies, findings, and theoretical frameworks across multiple papers for their research work.

## Task 2: Propose a Solution

### Solution Proposal (6 points)
Create an intelligent RAG (Retrieval Augmented Generation) system that converts PDF journal content into a searchable vector database, allowing researchers to ask natural language questions and receive accurate, contextually relevant answers with citations. The system will understand both the semantic meaning of queries and the academic context, providing more precise and useful results than traditional keyword search.

### Technology Stack (7 points)
- **Vector Database**: Qdrant - Chosen for its excellent performance with cosine similarity searches and robust concurrent processing capabilities. Run this locally on Docker.
- **Embeddings**: OpenAI text-embedding-3-small - Selected for its strong multilingual performance and cost-effectiveness
- **LLM**: GPT-3.5 Turbo - Offers good balance of performance and cost for academic text processing
- **Framework**: LangChain - Provides flexible RAG pipeline components and evaluation tools
- **Evaluation**: RAGAS - Comprehensive framework for measuring RAG system performance
- **Storage**: Local file system with embeddings cache - Ensures efficient reuse of processed embeddings

### Agent Usage (2 points)
The system uses agentic reasoning in the RAG pipeline (rag_graph.py) to:
1. Intelligently determine the most relevant context chunks for a given query
2. Synthesize information from multiple sources into coherent responses
3. Generate appropriate academic citations for sources used in responses

## Task 3: Dealing with the Data

### Data Sources and APIs (5 points)

I scraped articles from the following journals. You can see the scripts present in `scrape_ela.py` and `scrape_ling_mexicana.py`.

1. **ELA Journal PDFs**: Primary source of academic content, scraped from journal archives
2. **OpenAI API**: Used for both embeddings generation and text generation


### Chunking Strategy (5 points)
For this project, I used the standard `RecursiveTextSplitter` from LangChain to break up the journal articles into manageable chunks. This approach is simple and effective for initial prototyping, but it does not take into account the structure or metadata of the academic articles. As a result, the chunks are purely based on text length and do not always align with section boundaries or preserve citation context.

A better strategy would be to design a chunking process that stores each chunk along with rich metadata about the journal article—such as section titles, authors, and page numbers. This would make it easier to trace retrieved information back to its source and improve the quality of citations in generated answers.

In the future, I plan to experiment with semantic chunking methods that split the text based on meaning and document structure, rather than just length. This would help ensure that each chunk contains a coherent idea and that important academic context is preserved for retrieval and generation.


## Task 4: Building a Quick End-to-End Prototype (15 points)
The prototype has been successfully implemented with the following components:
1. PDF scraping and processing pipeline
2. Embedding generation and caching system
3. Qdrant vector database integration
4. RAG inference pipeline with LangChain
5. Local deployment with Docker for the vector database
6. Command-line interface for system interaction

## Task 5: Creating a Golden Test Data Set
This is done in the first script (`01_chunk_generate_embeddings_and_golden_test_set`). You can see the file goden_test_dataset.csv. After creating the golden test set (`golden_test_dataset.csv`), I ran an baselining epxeriment in ragas:

### RAGAS Framework Results (10 points)
Initial baseline evaluation metrics:
| Metric | Score | Performance |
|--------|--------|-------------|
| Context Recall | 0.534 | Moderate |
| Answer Relevancy | 0.597 | Decent |
| Faithfulness | 0.337 | Poor |
| Factual Correctness | 0.207 | Very Poor |
| Context Entity Recall | 0.011 | Critical |
| Noise Sensitivity | 0.012 | Critical |

### Performance Analysis (5 points)
The initial evaluation reveals several critical insights:
1. The system shows moderate success in finding relevant information (Context Recall: 0.534)
2. Responses are generally on-topic but could be more precise (Answer Relevancy: 0.597)
3. Major concerns with faithfulness (0.337) and factual correctness (0.207) indicate potential hallucination issues
4. Critical problems with entity recall (0.011) suggest the system struggles with technical terminology
5. Poor noise sensitivity (0.012) indicates issues with distinguishing relevant from irrelevant content

## Task 6: Advanced Retrieval (5 points)
I implemented a MultiQueryRetrieval strategy (you can see in the last commit)
This led to significant improvements:
- Context Recall improved by 21% (0.646 vs 0.534)
- Noise Sensitivity improved dramatically (0.052 vs 0.012)
- Factual Correctness improved by 34% (0.278 vs 0.207)

## Task 7: Assessing Performance

### Performance Comparison (5 points)
Advanced retrieval methods showed mixed results:

| Metric | Advanced | Baseline | Change |
|--------|----------|----------|---------|
| Context Recall | 0.646 | 0.534 | +21% |
| Answer Relevancy | 0.633 | 0.597 | +6% |
| Faithfulness | 0.288 | 0.337 | -15% |
| Factual Correctness | 0.278 | 0.207 | +34% |
| Entity Recall | 0.006 | 0.011 | -45% |
| Noise Sensitivity | 0.052 | 0.012 | +333% |

### Future Improvements (5 points)
Planned improvements for the second half of the course:
1. Implement specialized entity recognition for linguistics terminology
2. Develop a hybrid retrieval system combining semantic and keyword search
3. Add cross-lingual capabilities for English-Spanish translation
4. Improve context window optimization to enhance faithfulness
5. Implement citation validation to ensure accurate source attribution
6. Create a web-based user interface for easier system interaction
7. Add user feedback mechanisms to continuously improve retrieval quality

