import pandas as pd
import numpy as np
from ragas import evaluate
from ragas.metrics import (
    LLMContextRecall,
    Faithfulness,
    FactualCorrectness,
    ResponseRelevancy,
    ContextEntityRecall,
    NoiseSensitivity
)
from ragas import EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from ragas import RunConfig
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def load_evaluation_data(file_path: str = "golden_data_set_experiment_1.csv") -> pd.DataFrame:
    """Load the evaluation dataset."""
    df = pd.read_csv(file_path)
    
    # Set TEST_MODE to True to only process first 5 rows while testing
    TEST_MODE = False  # Comment this line to process all rows
    if TEST_MODE:
        print("Running in TEST MODE - only processing first 5 rows")
        return df.head(5)
    
    return df

def prepare_ragas_dataset(df: pd.DataFrame) -> EvaluationDataset:
    """Convert pandas DataFrame to RAGAS EvaluationDataset."""
    # Keep the original column names that RAGAS expects
    df_ragas = df.copy()
    
    # Convert string representation of lists to actual lists
    df_ragas['retrieved_contexts'] = df_ragas['retrieved_contexts'].apply(eval)
    df_ragas['reference_contexts'] = df_ragas['reference_contexts'].apply(eval)
    
    return EvaluationDataset.from_pandas(df_ragas)

def run_ragas_evaluation(dataset: EvaluationDataset) -> dict:
    """Run RAGAS evaluation with multiple metrics."""
    # Initialize evaluator LLM (using a cheaper model)
    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, max_tokens=4000)
    )
    
    # Configure metrics
    metrics = [
        LLMContextRecall(),
        Faithfulness(),
        FactualCorrectness(),
        ResponseRelevancy(),
        ContextEntityRecall(),
        NoiseSensitivity()
    ]
    
    # Set evaluation configuration
    custom_run_config = RunConfig(timeout=360)
    
    # Run evaluation
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        run_config=custom_run_config
    )
    
    # Convert EvaluationResult to dictionary
    return {
        metric.name: score
        for metric, score in zip(metrics, result.scores)
    }

def main():
    print("Loading evaluation data...")
    df = load_evaluation_data()
    print(f"Loaded {len(df)} evaluation records")
    
    print("\nPreparing RAGAS dataset...")
    dataset = prepare_ragas_dataset(df)
    
    print("\nRunning RAGAS evaluation...")
    results = run_ragas_evaluation(dataset)
    
    # Print results
    print("\nEvaluation Results:")
    print("==================")
    for metric_name, metric_scores in results.items():
        print(f"{metric_name}:")
        for score_name, score in metric_scores.items():
            if isinstance(score, (int, float, np.floating)):
                print(f"  - {score_name}: {float(score):.3f}")
            else:
                print(f"  - {score_name}: {score}")
        print()
    
    # Save results to CSV
    # Create a single row with the main scores
    flat_results = {}
    for metric_name, metric_scores in results.items():
        # Get the main score for each metric (usually matches the metric name)
        for score_name, score in metric_scores.items():
            if score_name == metric_name or score_name == f"{metric_name}(mode=f1)" or score_name == f"{metric_name}(mode=relevant)":
                flat_results[metric_name] = float(score) if not pd.isna(score) else 0.0
                break
    
    results_df = pd.DataFrame([flat_results])
    output_file = "evaluation_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()