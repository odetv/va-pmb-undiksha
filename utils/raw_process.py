import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from llm import EMBEDDER


# Check Directory Datasets & Vector Database
if not os.path.exists("src/datasets"):
    os.makedirs("src/datasets")
if not os.path.exists("src/vectordb"):
    os.makedirs("src/vectordb")


# 1. Process Load Documents
loader = PyPDFDirectoryLoader("src/datasets")
documents = loader.load()


# 2. Process Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=100,
    separators=[" ", ",", ".", "\n", "\n\n", "\n\n\n", "\f"]
)
chunks = text_splitter.split_documents(documents)


# 3. Process Embeddings
vectordb = FAISS.from_documents(chunks, EMBEDDER)
vectordb.save_local("src/vectordb")