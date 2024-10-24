import streamlit as st
import re
import sys
import os
import pandas as pd
import pdfplumber
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import build_graph
from langchain.schema.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv


load_dotenv()
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
openai_api_key = os.getenv("OPENAI_API_KEY")

CHUNK_SIZE = 900
CHUNK_OVERLAP = 100
VECTOR_PATH = "src/vectordb"
DATASET_PATH = "src/datasets"
MODEL_EMBEDDING = "text-embedding-3-small"
EMBEDDER = OpenAIEmbeddings(api_key=openai_api_key, model=MODEL_EMBEDDING)

def title_desc():
    st.set_page_config(page_title="VA PMB Undiksha")
    st.sidebar.title("Virtual Assistant PMB Undiksha")
    st.sidebar.image("assets\images\logo.webp", use_column_width=True)
    st.sidebar.write("Selamat datang di Virtual Assistant PMB Undiksha! Kami siap membantu Anda, silahkan bertanya😊")
    with st.sidebar:
        "[Source Code](https://github.com/odetv/va-pmb-undiksha)"
        "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new/odetv/va-pmb-undiksha?quickstart=1)"
    st.caption("Virtual Assistant Penerimaan Mahasiswa Baru Undiksha")


def raw_data():
    if st.button("Jalankan Semua Proses Data"):
    
        with st.expander("Proses Load Dokumen", expanded=False):
            try:
                with st.spinner("Sedang memproses dokumen, harap tunggu..."):
                    documents = []
                    DATASET_PATH = "src/datasets"
                    for file_name in os.listdir(DATASET_PATH):
                        if file_name.endswith('.pdf'):
                            documents.append(file_name)
                    if documents:
                        st.write("Daftar Dokumen yang Siap Diproses:")
                        document_df = pd.DataFrame({"No": range(1, len(documents) + 1), "Nama Dokumen": documents})
                        st.table(document_df)
                        st.success("Proses load dokumen selesai", icon="✅")
                    else:
                        st.warning("Tidak ada dokumen yang ditemukan untuk diproses.")
            except Exception as e:
                st.error(f"Terjadi kesalahan pada proses Load Dokumen: {str(e)}", icon="❌")

        with st.expander("Proses Chunking", expanded=False):
            try:
                with st.spinner("Sedang memproses chunking, harap tunggu..."):
                    documents = []
                    DATASET_PATH = "src/datasets"

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

                    CHUNK_SIZE = 900
                    CHUNK_OVERLAP = 100
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=CHUNK_SIZE,
                        chunk_overlap=CHUNK_OVERLAP,
                        length_function=len,
                        separators=[" "]
                    )
                    chunks = text_splitter.split_documents(documents)

                    chunk_data = []
                    for i, chunk in enumerate(chunks):
                        overlap_text = ""
                        if i > 0:
                            overlap_text = chunks[i-1].page_content[-CHUNK_OVERLAP:]

                        chunk_data.append({
                            "No Chunk": i + 1,
                            "Isi Chunk": chunk.page_content,
                            "Overlap": overlap_text
                        })

                    chunk_df = pd.DataFrame(chunk_data)
                    st.table(chunk_df)
                    st.success("Proses chunking selesai", icon="✅")
            except Exception as e:
                st.error(f"Terjadi kesalahan pada proses Chunking: {str(e)}", icon="❌")

        with st.expander("Proses Embeddings", expanded=False):
            try:
                with st.spinner("Sedang memproses embeddings untuk 5 chunk pertama, harap tunggu..."):
                    chunk_embeddings = []
                    chunks_to_embed = chunks[:5]
                    
                    for i, chunk in enumerate(chunks_to_embed):
                        embedding = EMBEDDER.embed_documents([chunk.page_content])[0]
                        chunk_embeddings.append({
                            "No Chunk": i + 1,
                            "Isi Chunk": chunk.page_content,
                            "Embedding": embedding
                        })
                        st.write(f"Embeddings untuk chunk {i + 1} berhasil dihitung.")

                    embedding_df = pd.DataFrame(chunk_embeddings)

                    embedding_df['Embedding'] = embedding_df['Embedding'].apply(lambda x: str(x[:5]) + " ...")  # Menampilkan sebagian embedding
                    
                    st.table(embedding_df)

                    st.success("Proses embeddings selesai untuk 5 chunk pertama", icon="✅")
            except Exception as e:
                st.error(f"Terjadi kesalahan pada proses Embeddings: {str(e)}", icon="❌")


def main():
    title_desc()
    raw_data()

if __name__ == "__main__":
    main()