import os
import pdfplumber
from langchain_openai import OpenAIEmbeddings
from langchain.schema.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv


load_dotenv()
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
openai_api_key = os.getenv("OPENAI_API_KEY")

CHUNK_SIZE = 900
CHUNK_OVERLAP = 100
VECTOR_PATH = "vectordb"
DATASET_PATH = "assets/datasets"
MODEL_EMBEDDING = "text-embedding-3-small"
EMBEDDER = OpenAIEmbeddings(api_key=openai_api_key, model=MODEL_EMBEDDING)

documents = []
for file_name in os.listdir(DATASET_PATH):
    if file_name.endswith('.pdf'):
        file_path = os.path.join(DATASET_PATH, file_name)
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    documents.append(
                        Document(page_content=text, metadata={"source": file_name})
                    )

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents(documents)
print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

vectordb = FAISS.from_documents(chunks, EMBEDDER)
vectordb.save_local(VECTOR_PATH)