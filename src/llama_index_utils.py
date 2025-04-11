from pinecone import Pinecone, ServerlessSpec
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from src.config import Config
from llama_index.core import settings
from llama_index.core.node_parser import SentenceSplitter

def create_index(docs):
    """
    Parse the documents into nodes, then create a LlamaIndex VectorStoreIndex
    backed by Pinecone for vector storage.
    """
    # 1. Parse docs into nodes using SimpleNodeParser
    settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
    nodes = parser.get_nodes_from_documents(docs)

    # 2. Initialize Pinecone client
    pc = Pinecone(api_key=Config.PINECONE_API_KEY)

    # 3. Create index if it doesn't exist (optional)
    if "examprepindex" not in pc.list_indexes().names():
        pc.create_index(
            name="examprepindex",
            dimension=384,  # adjust based on your embedding model
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    # 4. Connect to the Pinecone index
    pinecone_index = pc.Index("examprepindex")

    # 5. Create PineconeVectorStore
    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index,
        namespace="examprepnamespace"
    )

    # 6. Create a StorageContext using that Pinecone vector store
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 7. Build the VectorStoreIndex
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=HuggingFaceEmbedding(
            model_name=Config.EMBEDDING_MODEL,
            token=Config.HF_TOKEN
        )
    )

    return index
