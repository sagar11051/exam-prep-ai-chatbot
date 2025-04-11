from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from src.llm import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.prompts import PromptTemplate


def build_query_engine(index: VectorStoreIndex):
    # Set global Settings instead of using ServiceContext
    Settings.llm = HuggingFaceLLM()
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
    Settings.num_output = 512
    Settings.context_window = 3900

    retriever = VectorIndexRetriever(index=index, similarity_top_k=4)

    # Build the retriever and query engine (no need to pass service_context)
    QA_TEMPLATE = PromptTemplate(
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context, answer the question: {query_str}"
    )

    # Build query engine with custom prompt
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        text_qa_template=QA_TEMPLATE
    )

    return query_engine

