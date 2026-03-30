import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama


# -------------------------------
# CONFIG
# -------------------------------
PDF_PATH = "data/motion_transfer.pdf"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "pdf_collection"


# -------------------------------
# LOAD + SPLIT
# -------------------------------
def load_and_split(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    texts = splitter.split_documents(docs)

    print(f"Loaded {len(docs)} pages")
    print(f"Created {len(texts)} chunks")

    return texts


# -------------------------------
# VECTOR STORE
# -------------------------------
def create_vector_store(texts):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vector_store = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR
    )

    return vector_store


# -------------------------------
# RAG PIPELINE
# -------------------------------
def rag_query(vector_store, query):
    # Retrieve
    docs = vector_store.similarity_search(query, k=3)

    context = "\n\n".join(doc.page_content for doc in docs)

    # LLM
    llm = ChatOllama(
        model="llama3",
        temperature=0
    )

    prompt = f"""
You are an AI assistant.

Answer ONLY using the provided context.
If answer is not found, say "Not found in document".

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content


# -------------------------------
# MAIN
# -------------------------------
def main():
    texts = load_and_split(PDF_PATH)

    vector_store = create_vector_store(texts)

    query = "What are the limitations and future work mentioned by the authors?"

    answer = rag_query(vector_store, query)

    print("\n=== FINAL ANSWER ===\n")
    print(answer)


if __name__ == "__main__":
    main()