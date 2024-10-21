import streamlit as st
import re
from main import build_graph
from utils.llm import build_vector
import os
import pandas as pd
import pdfplumber
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
VECTOR_PATH = "vectordb"
DATASET_PATH = "assets/datasets"
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


def show_example_questions():
    example_questions = [
        "Bagaimana cara mendaftar?",
        "Jalur masuk yang tersedia?",
        "Kapan jadwal Tes SNBP?",
        "Dimana lokasi Undiksha?"
    ]
    cols = st.columns(4)
    for i, prompt in enumerate(example_questions):
        with cols[i]:
            if st.button(prompt):
                st.session_state.user_question = prompt
                response = build_graph(prompt)


def raw_data():
    if st.button("Jalankan Semua Proses Data"):
    
        with st.expander("Proses Load Dokumen", expanded=False):
            try:
                with st.spinner("Sedang memproses dokumen, harap tunggu..."):
                    documents = []
                    DATASET_PATH = "assets/datasets"
                    for file_name in os.listdir(DATASET_PATH):
                        if file_name.endswith('.pdf'):
                            documents.append(file_name)
                    if documents:
                        st.write("Daftar Dokumen yang Siap Diproses:")
                        document_df = pd.DataFrame({"No": range(1, len(documents) + 1), "Nama Dokumen": documents})
                        st.dataframe(document_df)
                        st.success("Proses load dokumen selesai", icon="✅")
                    else:
                        st.warning("Tidak ada dokumen yang ditemukan untuk diproses.")
            except Exception as e:
                st.error(f"Terjadi kesalahan pada proses Load Dokumen: {str(e)}", icon="❌")

        with st.expander("Proses Chunking", expanded=False):
            try:
                with st.spinner("Sedang memproses chunking, harap tunggu..."):
                    documents = []
                    DATASET_PATH = "assets/datasets"

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
                        separators=["\n\n", "\n", " ", ""]
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
                    st.dataframe(chunk_df)
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
                    
                    st.dataframe(embedding_df)

                    st.success("Proses embeddings selesai untuk 5 chunk pertama", icon="✅")
            except Exception as e:
                st.error(f"Terjadi kesalahan pada proses Embeddings: {str(e)}", icon="❌")


def main():
    title_desc()
    raw_data()

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Salam Harmoni🙏 Ada yang bisa saya bantu?", "raw_content": "Salam Harmoni🙏 Ada yang bisa saya bantu?", "images": []}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["raw_content"])
        if "images" in msg and msg["images"]:
            for img_url in msg["images"]:
                st.image(img_url, use_column_width=True)

    if prompt := st.chat_input("Ketik pertanyaan Anda di sini..."):
        st.session_state.messages.append({"role": "user", "content": prompt, "raw_content": prompt, "images": []})
        st.chat_message("user").write(prompt)
        with st.spinner("Sedang memproses, harap tunggu..."):
            response = build_graph(prompt)
            msg = re.sub(
                r'(https://aka\.undiksha\.ac\.id/api/ktm/generate/\S*)', 
                r'[Preview URL](\1)',
                response
            )
        raw_msg = msg
        html_msg = re.sub(
            r'(https://aka\.undiksha\.ac\.id/api/ktm/generate/\S*)', 
            r'<a href="\1" target="_blank">[Preview URL]</a>', 
            response
        )
        image_links = re.findall(r'(https?://\S+)', msg)
        images = []
        for link in image_links:
            if "https://aka.undiksha.ac.id/api/ktm/generate" in link or link.endswith(('.jpg', '.jpeg', '.png', '.gif')):
                images.append(link)

        st.session_state.messages.append({"role": "assistant", "content": html_msg, "raw_content": raw_msg, "images": images})
        st.chat_message("assistant").markdown(raw_msg)

        for img_url in images:
            st.image(img_url, use_column_width=True)


if __name__ == "__main__":
    main()