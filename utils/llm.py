import os
import shutil
import pdfplumber
import hashlib
import json
from langchain_ollama import OllamaLLM
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_groq import ChatGroq
from langchain.schema.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv


load_dotenv()
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
openai_api_key = os.getenv("OPENAI_API_KEY")


def chat_ollama(question: str, model = "gemma2"):
    ollama = OllamaLLM(base_url=ollama_base_url, model=model, verbose=True)
    result = ollama.invoke(question)
    return result


def chat_openai(question: str):
    openai = ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini")
    result = openai.invoke(question).content if hasattr(openai.invoke(question), "content") else openai.invoke(question)
    return result


def chat_groq(question: str):
    groq = ChatGroq(
        model="gemma2-9b-it",
        max_tokens=None,
        timeout=None,
    )
    result = groq.invoke(question).content if hasattr(groq.invoke(question), "content") else groq.invoke(question)
    return result


def embedding_openai():
    MODEL_EMBEDDING = "text-embedding-3-small"
    # MODEL_EMBEDDING = "text-embedding-3-large"
    EMBEDDER = OpenAIEmbeddings(api_key=openai_api_key, model=MODEL_EMBEDDING)
    return MODEL_EMBEDDING, EMBEDDER


def embedding_ollama():
    MODEL_EMBEDDING = "bge-m3"
    EMBEDDER = OllamaEmbeddings(base_url=ollama_base_url, model=MODEL_EMBEDDING, show_progress=True)
    return MODEL_EMBEDDING, EMBEDDER
MODEL_EMBEDDING, EMBEDDER = embedding_openai()


def build_vector():
    CHUNK_SIZE = 900
    CHUNK_OVERLAP = 100
    VECTOR_PATH = "vectordb"
    DATASET_PATH = "assets/datasets"
    HASH_FILE = "utils/logfile/file_hashes.json"
    PARAM_FILE = "utils/logfile/file_params.json"

    if not os.path.exists('utils/logfile'):
        os.makedirs('utils/logfile')

    # Fungsi untuk menghitung hash MD5 dari file yang diberikan
    def calculate_md5(file_path):
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    # Fungsi untuk memuat hash file yang sudah ada dari file HASH_FILE
    def load_hashes():
        if os.path.exists(HASH_FILE):
            with open(HASH_FILE, 'r') as f:
                return json.load(f)
        return {}


    # Fungsi untuk menyimpan hash file yang baru ke dalam HASH_FILE
    def save_hashes(hashes):
        with open(HASH_FILE, 'w') as f:
            json.dump(hashes, f)


    # Fungsi untuk memuat parameter yang sudah ada dari file PARAM_FILE
    def load_params():
        if os.path.exists(PARAM_FILE):
            with open(PARAM_FILE, 'r') as f:
                return json.load(f)
        return {}


    # Fungsi untuk menyimpan parameter yang baru ke dalam PARAM_FILE
    def save_params(params):
        with open(PARAM_FILE, 'w') as f:
            json.dump(params, f)


    hashes = load_hashes()
    prev_params = load_params()
    new_params = {
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "model_embedding": MODEL_EMBEDDING
    }

    # Menentukan apakah perlu membangun ulang vektor DB berdasarkan perubahan file atau parameter
    need_rebuild = not os.path.exists(VECTOR_PATH) or prev_params != new_params

    documents = [] # Daftar untuk menyimpan dokumen yang diproses
    new_hashes = {} # Tempat untuk menyimpan hash yang baru

    for file_name in os.listdir(DATASET_PATH):
        if file_name.endswith('.pdf'):
            file_path = os.path.join(DATASET_PATH, file_name)
            file_hash = calculate_md5(file_path)
            new_hashes[file_name] = file_hash
            if hashes.get(file_name) != file_hash:
                print(f"File changed: {file_name}")
                need_rebuild = True
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            documents.append(
                                Document(page_content=text, metadata={"source": file_name})
                            )
                            
    save_hashes(new_hashes) # Menyimpan hash yang baru
    save_params(new_params) # Menyimpan parameter yang baru

    if need_rebuild:
        if not documents:
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

        if chunks:
            if os.path.exists(VECTOR_PATH):
                shutil.rmtree(VECTOR_PATH)
            vectordb = FAISS.from_documents(chunks, EMBEDDER)
            vectordb.save_local(VECTOR_PATH)
            
            print(f"Saved {len(chunks)} chunks to {VECTOR_PATH}.")
        else:
            print("No valid chunks to update in VectorDB.")
    else:
        print("No changes in files or parameters, skipping VectorDB update.")
    return VECTOR_PATH