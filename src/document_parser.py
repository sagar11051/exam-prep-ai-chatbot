import pdfplumber
from llama_index.core.schema import Document

def parse_documents(uploaded_files):
    docs = []

    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name

        if file_name.endswith(".txt"):
            content = uploaded_file.read().decode("utf-8", errors="ignore")

        elif file_name.endswith(".pdf"):
            content = ""
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        content += text + "\n"

        else:
            print(f"[WARNING] Unsupported file type: {file_name}")
            continue

        if content.strip():
            docs.append(Document(text=content.strip(), doc_id=file_name))

    return docs
