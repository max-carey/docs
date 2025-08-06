import pandas as pd
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
from dotenv import load_dotenv

load_dotenv()

def load_evaluation_data(file_path: str = "golden_data_set_experiment_3.csv") -> pd.DataFrame:
    """Load the evaluation dataset."""
    df = pd.read_csv(file_path)
    

    TEST_MODE = False
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
        ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
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
    
    # Return the results
    return result

def main():
    df = load_evaluation_data()
    dataset = prepare_ragas_dataset(df)
    results = run_ragas_evaluation(dataset)
    results_pandas = results.to_pandas()
    mean_scores = results_pandas.mean(numeric_only=True)
    print("\nMean Scores:")
    print("===========")
    for metric_name, score in mean_scores.items():
        print(f"{metric_name}: {score:.3f}")

if __name__ == "__main__":
    main()