import patch_streamlit_watcher
import streamlit as st
from src.config import Config
from src.document_parser import parse_documents
from src.llama_index_utils import create_index
from src.retrieval import build_query_engine

st.set_page_config(page_title="Exam Prep AI Chatbot", layout="wide")
st.title("🧠 Exam Preparation AI Chatbot")

# Cache index so it's not re-created on every rerun
@st.cache_resource(show_spinner="🔄 Creating or fetching index...")
def get_index(_docs):
    return create_index(_docs)

uploaded_files = st.file_uploader(
    "📚 Upload your study documents",
    accept_multiple_files=True,
    type=["pdf", "txt", "md"]
)

if uploaded_files:
    with st.spinner("📄 Parsing documents..."):
        docs = parse_documents(uploaded_files)
        st.success("✅ Documents parsed successfully!")

    with st.spinner("📦 Creating/updating index in Pinecone..."):
        index = get_index(docs)
       ## index = get_index(docs)
        st.success("✅ Index creation done!")

    query_str = st.text_input("💬 Ask a question about your uploaded content")

    if st.button("Get Answer"):
        if query_str.strip() == "":
            st.warning("Please enter a question.")
        else:
            with st.spinner("🤖 Generating answer..."):
                # Use the build_query_engine function which internally creates HuggingFaceLLM
                query_engine = build_query_engine(index)
                response = query_engine.query(query_str)

            st.subheader("Answer")
            # Use .response if available, otherwise .text; adjust based on your response object structure.
            st.write(response.response)
            with st.expander("🔍 Retrieved Context"):
                for node in response.source_nodes:
                    st.markdown(f"**Score:** {node.score:.2f}")
                    st.write(node.node.text)
else:
    st.info("📥 Please upload at least one document to begin.")
