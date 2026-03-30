# RAG (Retrieval Augmented Generation) System

## 📚 Table of Contents
- What is RAG?
- RAG Architecture Overview
- Project Data Flow
- Detailed Process
- Setup & Dependencies

## What is RAG?

**Retrieval-Augmented Generation (RAG)** is a way to make AI answers more reliable by combining searching for relevant information and then generating a response. Instead of guessing based only on old training data, it first finds useful data from external sources (like documents or databases) and then uses it to give a better answer.

### Why RAG?
- **Accurate Responses**: Ground LLM answers in actual document content
- **Reduced Hallucinations**: Prevents the model from making up information
- **Domain-Specific Knowledge**: Augments LLMs with specialized information
- **Up-to-date Information**: Leverages your documents rather than relying on training data

## 📊 RAG Data Flow Diagram

<p align="center">
  <img src="https://github.com/user-attachments/assets/c5d66bfe-72f1-4fb8-a63a-2268f0e4ecd3" width="600"/>
</p>

### Complete Implementation Flow of Code

#### 1. **Load PDF**
```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("Generic Motion Transfer from Video to Static Image Using Deep Learning Techniques.pdf")
docs = loader.load()
```

#### 2. **Text into Chunks**
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)

texts = text_splitter.split_documents(docs)
```

#### 3. **Create Embeddings**
```python
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")
```

#### 4. **Store in Vector DB**
```python
from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="pdf_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db"
)

ids = vector_store.add_documents(documents=texts)
```

#### 5. **Retrieve & Augment**
```python
query = "What are the limitations and future work mentioned by the authors?"

# Retrieval
retrieved_docs = vector_store.similarity_search(query, k=3)

# Augmentation
context = "\n\n".join(doc.page_content for doc in retrieved_docs)
```

#### 6. **Generate Answer**
```python
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3",
    temperature=0,
    validate_model_on_init=True
)

prompt = f"""
Answer the question based ONLY on the context below.

Context:
{context}

Question:
{query}
"""

response = llm.invoke(prompt)
print(response.content)
```

## Setup & Dependencies

### Required Packages
```bash
pip install langchain
pip install langchain-community
pip install langchain-chroma
pip install chromadb
pip install langchain-ollama
pip install pypdf
```

### Required Models (Ollama)
```bash
ollama pull nomic-embed-text  # Embedding model
ollama pull llama3             # LLM model
```

### Directory Structure
```
RAG/
├── main.ipynb                    # Main implementation
├── README.md                     # This file
├── chroma_langchain_db/          # Vector database storage
│   ├── chroma.sqlite3
│   └── [collection folders]
└── Generic Motion Transfer.pdf   # Input document
```
## Key Concepts

| Term | Definition |
|------|-----------|
| **Embedding** | Numerical vector representation of text capturing semantic meaning |
| **Vector DB** | Database optimized for storing and searching high-dimensional vectors |
| **Similarity Search** | Finding most similar vectors to a query vector |
| **Chunk Overlap** | Repeating text boundaries to preserve context between chunks |
| **Augmentation** | Combining user query with retrieved context |
| **LLM** | Large Language Model that generates responses based on augmented prompts |
