import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from llm import EMBEDDER


load_dotenv()
DATASETS_DIR = os.getenv("APP_DATASETS_DIR")
VECTORDB_DIR = os.getenv("APP_VECTORDB_DIR")


# Check Directory Datasets & Vector Database
if not os.path.exists(DATASETS_DIR):
    os.makedirs(DATASETS_DIR)
if not os.path.exists(VECTORDB_DIR):
    os.makedirs(VECTORDB_DIR)


# 1. Process Load Documents
loader = PyPDFDirectoryLoader(DATASETS_DIR)
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
vectordb.save_local(VECTORDB_DIR)