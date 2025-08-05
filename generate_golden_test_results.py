import pandas as pd
from rag_graph import InferenceEngine, vectorstore
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def load_golden_dataset(file_path: str = "golden_test_dataset.csv") -> pd.DataFrame:
    """Load the golden test dataset from CSV."""
    return pd.read_csv(file_path)

def run_rag_evaluation(dataset: pd.DataFrame) -> pd.DataFrame:
    """Run RAG system on each test case and collect results."""
    # Initialize LLM and inference engine
    llm = ChatOpenAI(
        model="gpt-3.5-turbo-0125",
        temperature=0,
    )
    engine = InferenceEngine(vectorstore, llm)
    
    # Initialize result lists
    responses = []
    retrieved_contexts = []
    
    # Process each test case
    for _, row in dataset.iterrows():
        try:
            print(f"\nProcessing query: {row['user_input']}")
            
            # Get retrieved contexts
            contexts = engine.retrieve(row['user_input'])
            retrieved_contexts.append(contexts)
            
            # Get response
            response = engine.run_inference(row['user_input'])
            responses.append(response)
            
            print(f"Response generated: {response[:100]}...")
        except Exception as e:
            print(f"Error processing query '{row['user_input']}': {str(e)}")
            retrieved_contexts.append([])
            responses.append("")
    
    # Add results to dataframe
    dataset['retrieved_contexts'] = retrieved_contexts
    dataset['response'] = responses
    
    return dataset

def save_results(results: pd.DataFrame, output_file: str = "golden_data_set_experiment_1.csv"):
    """Save evaluation results to CSV."""
    results.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

def main():
    print("Loading golden dataset...")
    dataset = load_golden_dataset()
    
    print(f"Loaded {len(dataset)} test cases")
    
    print("\nRunning RAG evaluation...")
    results = run_rag_evaluation(dataset)
    
    print("\nSaving results...")
    save_results(results)
    
    print("Evaluation complete!")

if __name__ == "__main__":
    main()