from typing import TypedDict, Annotated, List, Dict, Union
from typing_extensions import TypedDict
from operator import add
from dataclasses import dataclass
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient, models
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Qdrant client with local storage
client = QdrantClient(url="http://localhost:6333")

# Create collection if it doesn't exist
COLLECTION_NAME = "linguistics_articles"
try:
    client.get_collection(COLLECTION_NAME)
except ValueError:
    # Create new collection
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=1536,  # OpenAI embedding dimensions
            distance=models.Distance.COSINE
        ),
    )

# Initialize embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=1536
)

# Initialize vector store
vectorstore = Qdrant(
    client=client,
    collection_name=COLLECTION_NAME,
    embeddings=embeddings,
)

# Define our state type
class AgentState(TypedDict):
    messages: Annotated[List[Union[HumanMessage, AIMessage]], add]
    context: List[str]
    current_step: str

@dataclass
class PromptConstructor:
    """Class responsible for constructing and managing prompts."""
    
    def get_system_message(self) -> str:
        """Returns the system message template."""
        return """You are a helpful assistant answering questions about linguistics articles. 
        Use the following context to answer the user's question:
        
        {context}"""

    def create_prompt_template(self) -> ChatPromptTemplate:
        """Creates and returns the chat prompt template."""
        print("\n=== Creating Prompt Template ===")
        template = ChatPromptTemplate.from_messages([
            ("system", self.get_system_message()),
            ("human", "{question}")
        ])
        print("Prompt template created successfully")
        return template

class InferenceEngine:
    """Class responsible for running inference and managing the retrieval process."""
    
    def __init__(self, vectorstore: Qdrant, llm: ChatOpenAI):
        self.vectorstore = vectorstore
        self.llm = llm
        self.prompt_constructor = PromptConstructor()

    def retrieve(self, query: str) -> List[str]:
        """Retrieves relevant documents for the given query."""
        try:
            # Use the raw Qdrant client to get results with payloads
            results = self.vectorstore.client.search(
                collection_name=self.vectorstore.collection_name,
                query_vector=self.vectorstore.embeddings.embed_query(query),
                limit=3
            )
            
            contents = []
            for i, result in enumerate(results):
                print(f"\nResult {i + 1}:")
                print(f"Score: {result.score}")
                if result.payload:
                    print(f"Payload: {result.payload}")
                    if 'text' in result.payload:
                        contents.append(result.payload['text'])
                        print(f"Content: {result.payload['text'][:200]}...")
                    else:
                        print("No text in payload")
            
            print("Contents to return:", contents)
            return contents
        except Exception as e:
            print(f"Error during similarity search: {str(e)}")
            return []

    def format_context(self, context_docs: List[str]) -> str:
        """Formats the context documents into a single string."""
        print("\n=== Formatting Context ===")
        print(f"Number of context documents: {len(context_docs)}")
        print("context docs", context_docs)
        formatted_context = "\n\n".join(context_docs)
        
        print("formatted context", formatted_context)
        return formatted_context

    def create_chain(self):
        """Creates the RAG chain with explicit prompt construction and inference steps."""
        print("\n=== Creating RAG Chain ===")
        
        # Create the prompt template
        prompt = self.prompt_constructor.create_prompt_template()
        
        # Create the chain with explicit steps
        chain = {
            "context": lambda x: self.format_context(self.retrieve(x["question"])),
            "question": lambda x: x["question"]
        } | prompt | self.llm | StrOutputParser()
        
        print("Chain created successfully")
        return chain

    def run_inference(self, query: str) -> str:
        """Runs the complete inference process with detailed logging."""
        print("\n=== Starting Inference ===")
        print(f"Input query: {query}")
        
        chain = self.create_chain()
        
        try:
            print("\n=== Running Chain ===")
            response = chain.invoke({"question": query})
            print("\n=== Inference Complete ===")
            print(f"Response: {response}")
            return response
        except Exception as e:
            print(f"\n=== Inference Error ===")
            print(f"Error: {str(e)}")
            if "Collection linguistics_articles not found" in str(e):
                print("\nThe Qdrant collection is empty. Please run process_pdfs.py first to populate it with documents.")
            raise

def main():
    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-3.5-turbo-0125",
        temperature=0,
    )
    
    # Create inference engine
    engine = InferenceEngine(vectorstore, llm)
    
    # Example query
    query = "What is the UNAM?"
    
    try:
        response = engine.run_inference(query)
        print("\n=== Final Result ===")
        print(f"Query: {query}")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Failed to get response: {str(e)}")

if __name__ == "__main__":
    main()