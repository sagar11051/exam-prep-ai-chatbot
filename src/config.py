import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Hugging Face model name for LLM (e.g., "gpt2", "distilgpt2", or any other model)
    MODEL_NAME = "EleutherAI/gpt-neo-1.3B"
    
    # Pinecone configuration
    PINECONE_API_KEY = "pcsk_57N6zT_4rmpWmQmUEZnFBkqkvBLyqiA4xdRUPQ1196SBq19WoWKHVaoR5eCHaFXw1MEZKZ"
    PINECONE_ENV = "gcp-starter"
    
    # Example embedding model name if integrating HF embeddings with LlamaIndex
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
    INDEX_NAME = "exam_prep_index"
    NAMESPACE = "exam_prep_namespace"