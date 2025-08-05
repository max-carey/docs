import os
import logging
import asyncio
import time
from pathlib import Path
from typing import List, Tuple
from functools import partial

import pandas as pd
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

from shared_embeddings import save_document_embeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load environment variables
load_dotenv()

# Constants
PDF_DIRS = ["downloads/ela_issues", "downloads"]  # Both ELA and Linguistica Mexicana directories
MAX_DOCS_FOR_TESTING = None  # Set to None to use all documents
TESTSET_SIZE = 50  # Number of test cases to generate
INITIAL_BATCH_SIZE = 5  # Initial number of documents to process in parallel
INITIAL_CONCURRENT_TASKS = 5  # Initial number of concurrent async tasks
BATCH_SIZE = INITIAL_BATCH_SIZE  # Current batch size (will adjust based on rate limits)
MAX_CONCURRENT_TASKS = INITIAL_CONCURRENT_TASKS  # Current concurrency (will adjust based on rate limits)
SUCCESS_COUNT_FOR_SPEEDUP = 10  # Number of successful requests before trying to speed up

# Track successful requests
successful_requests = 0

def load_pdf(file_path: str) -> List[Document]:
    """Load PDF and return its pages as documents."""
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        
        if not pages:
            return []
            
        # Add source metadata
        for page in pages:
            page.metadata['source'] = file_path
            
        return pages
        
    except Exception as e:
        logging.error(f"Error loading PDF {file_path}: {str(e)}")
        return []

async def generate_embedding(doc: Document, generator_embeddings) -> Tuple[Document, List[float]]:
    """Generate embedding for a single document asynchronously."""
    global BATCH_SIZE, MAX_CONCURRENT_TASKS, successful_requests
    max_retries = 3
    retry_delay = 2.0
    
    for attempt in range(max_retries):
        try:
            # Run the embedding generation in a thread pool since it's a CPU-bound operation
            loop = asyncio.get_event_loop()
            vector = await loop.run_in_executor(
                None, 
                generator_embeddings.embed_query, 
                doc.page_content
            )
            
            # Track successful request and potentially speed up
            successful_requests += 1
            if successful_requests >= SUCCESS_COUNT_FOR_SPEEDUP:
                if BATCH_SIZE < INITIAL_BATCH_SIZE or MAX_CONCURRENT_TASKS < INITIAL_CONCURRENT_TASKS:
                    BATCH_SIZE = min(INITIAL_BATCH_SIZE, BATCH_SIZE * 2)
                    MAX_CONCURRENT_TASKS = min(INITIAL_CONCURRENT_TASKS, MAX_CONCURRENT_TASKS * 2)
                    logging.info(f"Processing running smoothly, increasing batch size to {BATCH_SIZE} and concurrency to {MAX_CONCURRENT_TASKS}")
                successful_requests = 0  # Reset counter
                
            return doc, vector
            
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                # Only slow down if we hit a rate limit
                logging.warning(f"Rate limit hit, retrying in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
                
                # Reduce batch size and concurrency after hitting rate limit
                BATCH_SIZE = max(1, BATCH_SIZE // 2)
                MAX_CONCURRENT_TASKS = max(1, MAX_CONCURRENT_TASKS // 2)
                successful_requests = 0  # Reset success counter
                logging.info(f"Reducing batch size to {BATCH_SIZE} and concurrency to {MAX_CONCURRENT_TASKS}")
            else:
                logging.error(f"Error generating embedding for document {doc.metadata.get('source', 'unknown')}: {str(e)}")
                raise

async def process_document_batch(batch: List[Document], generator_embeddings) -> List[Tuple[Document, List[float]]]:
    """Process a batch of documents to generate their embeddings asynchronously."""
    tasks = [generate_embedding(doc, generator_embeddings) for doc in batch]
    return await asyncio.gather(*tasks)

async def generate_embeddings_async(docs: List[Document], generator_embeddings) -> Tuple[List[Document], List[List[float]]]:
    """Generate embeddings for all documents using async batching."""
    processed_docs = []
    vectors = []
    
    # Process documents in batches
    batches = [docs[i:i + BATCH_SIZE] for i in range(0, len(docs), BATCH_SIZE)]
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
    
    async def process_batch_with_semaphore(batch):
        async with semaphore:
            return await process_document_batch(batch, generator_embeddings)
    
    # Process all batches with progress bar
    results = await tqdm_asyncio.gather(
        *[process_batch_with_semaphore(batch) for batch in batches],
        desc="Generating embeddings"
    )
    
    # Flatten results
    for batch_results in results:
        for doc, vector in batch_results:
            processed_docs.append(doc)
            vectors.append(vector)
    
    return processed_docs, vectors

async def generate_test_cases_async(generator: TestsetGenerator, docs: List[Document], testset_size: int) -> pd.DataFrame:
    """Generate test cases asynchronously."""
    loop = asyncio.get_event_loop()
    # Run the test generation in a thread pool since it's CPU-bound
    df = await loop.run_in_executor(
        None,
        partial(generator.generate_with_langchain_docs, docs, testset_size=testset_size)
    )
    return df.to_pandas()

async def generate_golden_dataset(docs: List[Document], output_file: str = "golden_test_dataset.csv", testset_size: int = TESTSET_SIZE, cache_key: str = "golden_test"):
    """Generate golden test dataset using Ragas and save it as CSV.
    Also saves document embeddings for reuse in the main processing.
    
    Args:
        docs: List of documents to generate test data from
        output_file: Path to save the CSV file
        testset_size: Number of test cases to generate
        cache_key: Identifier for caching embeddings
    
    Returns:
        pd.DataFrame: The generated test dataset
    """
    logging.info("Generating golden test dataset...")
    
    # Initialize models for test generation
    generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-3.5-turbo-0125"))
    generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
    
    # Initialize test generator with the models
    generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
    logging.info("Initialized Ragas test generator")
    
    # Use all documents if MAX_DOCS_FOR_TESTING is None, otherwise use the specified number
    if MAX_DOCS_FOR_TESTING is None:
        docs_subset = docs
        logging.info(f"Using all {len(docs_subset)} documents for test generation")
    else:
        docs_subset = docs[:MAX_DOCS_FOR_TESTING]
        logging.info(f"Using {len(docs_subset)} documents for test generation (out of {len(docs)} total documents)")
    
    # Generate embeddings asynchronously
    processed_docs, vectors = await generate_embeddings_async(docs_subset, generator_embeddings)
    
    # Save embeddings for reuse
    save_document_embeddings(processed_docs, vectors, cache_key)
    logging.info(f"Saved embeddings for {len(processed_docs)} documents")
    
    # Generate test dataset using latest Ragas API
    logging.info("Generating test cases...")
    try:
        df = await generate_test_cases_async(generator, processed_docs, testset_size)
        df.to_csv(output_file, index=False)
        logging.info(f"Successfully saved {len(df)} test cases to {output_file}")
        return df
    except Exception as e:
        logging.error(f"Error in Ragas test generation: {str(e)}")
        raise

def main():
    """Main function to generate golden test dataset."""
    # Collect all documents
    all_docs = []
    for pdf_dir in PDF_DIRS:
        # If MAX_DOCS_FOR_TESTING is not None, check if we've reached the limit
        if MAX_DOCS_FOR_TESTING is not None and len(all_docs) >= MAX_DOCS_FOR_TESTING:
            logging.info(f"Reached maximum number of documents ({MAX_DOCS_FOR_TESTING})")
            break
            
        pdf_path = Path(pdf_dir)
        if not pdf_path.exists():
            logging.warning(f"Directory not found: {pdf_dir}")
            continue
            
        for pdf_file in pdf_path.glob("*.pdf"):
            try:
                logging.info(f"Loading {pdf_file}")
                docs = load_pdf(str(pdf_file))
                if docs:
                    all_docs.extend(docs)
                    # If MAX_DOCS_FOR_TESTING is not None, check if we've reached the limit
                    if MAX_DOCS_FOR_TESTING is not None and len(all_docs) >= MAX_DOCS_FOR_TESTING:
                        logging.info(f"Reached maximum number of documents ({MAX_DOCS_FOR_TESTING})")
                        break
                    
            except Exception as e:
                logging.error(f"Error processing {pdf_file}: {str(e)}")
                continue
    
    if not all_docs:
        logging.error("No documents were loaded. Exiting.")
        return
    
    # Generate golden test dataset
    try:
        # Run the async parts using asyncio
        df = asyncio.run(generate_golden_dataset(all_docs))
        logging.info(f"Generated {len(df)} test cases")
        
        # Display a summary of the dataset
        logging.info("\nDataset Summary:")
        if len(df) > 0:
            logging.info(f"Average question length: {df['user_input'].str.len().mean():.1f} characters")
            logging.info(f"Average answer length: {df['reference'].str.len().mean():.1f} characters")
            logging.info(f"Average number of context chunks: {df['reference_contexts'].apply(len).mean():.1f}")
        else:
            logging.warning("No test cases were generated. This might indicate an issue with the test generation process.")
        
        logging.info("\nGolden test dataset generation completed successfully!")
    except Exception as e:
        logging.error(f"Error generating golden test dataset: {str(e)}")

if __name__ == "__main__":
    main()