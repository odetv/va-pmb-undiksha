import os
import streamlit as st
import pandas as pd
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
from utils.llm import chat_openai, EMBEDDER
from dotenv import load_dotenv


load_dotenv()
STREAMLIT_KEY_ADMIN = os.getenv("STREAMLIT_KEY_ADMIN")


def setup_page():
    st.set_page_config(layout="wide", page_title="Debug - VA PMB Undiksha", page_icon="assets/images/logo.png")
    st.sidebar.image("assets\images\logo.png", use_column_width=True)
    st.sidebar.title("Panel Simulasi Proses Data Virtual Assistant PMB Undiksha")
    with st.sidebar:
        "[Source Code](https://github.com/odetv/va-pmb-undiksha)"
        "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new/odetv/va-pmb-undiksha?quickstart=1)"


def load_documents():
    if "documents" not in st.session_state:
        progress = st.progress(0)

        if not os.path.exists("src/datasets"):
            os.makedirs("src/datasets")
        loader = PyPDFDirectoryLoader("src/datasets")
        documents = loader.load()

        file_data = [{"No": i + 1, "Nama File": doc.metadata["source"]} for i, doc in enumerate(documents)]
        df_files = pd.DataFrame(file_data)
        df_files = df_files.drop_duplicates(subset=["Nama File"], keep="first").reset_index(drop=True)
        df_files["No"] = range(1, len(df_files) + 1)
        st.session_state.documents = documents
        
        progress.progress(100)
        st.success("Dokumen telah berhasil disiapkan.", icon="✅")
        st.dataframe(df_files, use_container_width=True)
    else:
        st.success("Dokumen telah disiapkan sebelumnya.", icon="✅")
        documents = st.session_state.documents
        file_data = [{"No": i + 1, "Nama File": doc.metadata["source"]} for i, doc in enumerate(documents)]
        df_files = pd.DataFrame(file_data)
        df_files = df_files.drop_duplicates(subset=["Nama File"], keep="first").reset_index(drop=True)
        df_files["No"] = range(1, len(df_files) + 1)
        st.dataframe(df_files, use_container_width=True)
    return documents


def chunk_documents(documents):
    if "chunks" not in st.session_state:
        progress = st.progress(0)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=100,
            separators=[" ", ",", ".", "\n", "\n\n", "\n\n\n", "\f"]
        )
        chunks = text_splitter.split_documents(documents)
        chunk_data = []
        
        for i, chunk in enumerate(chunks):
            if i > 0:
                overlap = chunks[i].page_content[:90]
            else:
                overlap = ""
            chunk_data.append({"No Chunk": i + 1, "Isi Chunk": chunk.page_content, "Overlap": overlap})
            progress.progress((i + 1) / len(chunks))
        
        df_chunks = pd.DataFrame(chunk_data)
        st.success("Proses chunking dokumen berhasil.", icon="✅")
        st.table(df_chunks)
        st.session_state.chunks = chunks
    else:
        st.success("Dokumen telah di-chunking sebelumnya.", icon="✅")
        chunks = st.session_state.chunks
        chunk_data = [{"No Chunk": i + 1, "Isi Chunk": chunk.page_content, "Overlap": chunk.page_content[:90]} for i, chunk in enumerate(chunks)]
        df_chunks = pd.DataFrame(chunk_data)
        st.table(df_chunks)
    return chunks


def embeddings_documents(chunks):
    if "embeddings" not in st.session_state:
        progress = st.progress(0)

        if not os.path.exists("src/vectordb"):
            os.makedirs("src/vectordb")
        vectordb = FAISS.from_documents(chunks, EMBEDDER)
        vectordb.save_local("src/vectordb")

        embeddings_data = []

        for i, chunk in enumerate(chunks):
            chunk_text = chunk.page_content
            try:
                embedding = EMBEDDER.embed_query(chunk_text)
                embeddings_data.append({"No Chunk": i + 1, "Isi Chunk": chunk_text, "Embedding": embedding})
            except Exception as e:
                st.error(f"Error pada chunk {i+1}: {str(e)}")
            progress.progress((i + 1) / len(chunks))

        df_embeddings = pd.DataFrame(embeddings_data)
        st.success("Proses embedding dokumen berhasil.", icon="✅")
        st.dataframe(df_embeddings, use_container_width=True)
        st.session_state.embeddings = embeddings_data
    else:
        st.success("Embedding dokumen telah dilakukan sebelumnya.", icon="✅")
        embeddings_data = st.session_state.embeddings
        df_embeddings = pd.DataFrame(embeddings_data)
        st.dataframe(df_embeddings, use_container_width=True)
    return [data["Embedding"] for data in embeddings_data]


def entryDatasets():
    st.write("### • Simulasi Entry Datasets")
    st.caption("Siapkan datasets yang akan digunakan!")

    with st.expander("Upload Datasets", expanded=False):
        if not os.path.exists("src/datasets"):
            os.makedirs("src/datasets")
        if 'files' not in st.session_state:
            st.session_state.files = os.listdir("src/datasets")
        if 'upload_status' not in st.session_state:
            st.session_state.upload_status = ""
        if 'is_uploaded' not in st.session_state:
            st.session_state.is_uploaded = False

        uploaded_files = st.file_uploader("Upload multiple files", 
                                        type=["pdf", "txt", "docx", "doc"], 
                                        accept_multiple_files=True,
                                        key="file_uploader")

        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_path = os.path.join("src/datasets", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            st.session_state.upload_status = f"{len(uploaded_files)} file berhasil di-upload."
            st.session_state.files = os.listdir("src/datasets")
            st.session_state.is_uploaded = True

        if st.session_state.upload_status:
            st.success(st.session_state.upload_status)

    with st.expander("Daftar Datasets", expanded=False):
        current_files = os.listdir("src/datasets")
        if current_files:

            files_to_delete = []

            for file in current_files:
                is_checked = st.checkbox(file, key=file)

                if is_checked:
                    files_to_delete.append(file)

            if st.button("Hapus File yang Dipilih"):
                deleted_count = 0
                
                for file in files_to_delete:
                    file_path = os.path.join("src/datasets", file)
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            deleted_count += 1
                    except Exception as e:
                        st.error(f"Gagal menghapus file {file}: {str(e)}")

                if deleted_count > 0:
                    st.success(f"{deleted_count} file berhasil dihapus.")
                    st.session_state.files = os.listdir("src/datasets")
                    st.session_state.upload_status = ""
                    st.session_state.is_uploaded = False
                    st.rerun()




def raw_process():
    st.write("### • Simulasi Proses Data RAW")
    
    documents = None
    chunks = None
    st.caption("Tekan tombol dibawah untuk mulai simulasi proses!")
    start_raw_process = st.button("Mulai Proses")
    
    with st.expander("Proses Menyiapkan Dokumen", expanded=start_raw_process):
        if start_raw_process or "documents" in st.session_state:
            documents = load_documents()
        else:
            st.warning("Menunggu proses dimulai...", icon="⏳")

    with st.expander("Proses Chunking", expanded=(documents is not None)):
        if documents is not None or "chunks" in st.session_state:
            chunks = chunk_documents(documents)
        else:
            st.warning("Menunggu proses penyiapan dokumen...", icon="⏳")

    with st.expander("Proses Embeddings", expanded=(chunks is not None)):
        if chunks is not None or "embeddings" in st.session_state:
            vectordb = embeddings_documents(chunks)
        else:
            st.warning("Menunggu proses chunking...", icon="⏳")


def query_user_process():
    st.write("### • Simulasi Proses Query Pengguna")

    user_question = st.chat_input("Ketik pertanyaan Anda di sini...")
    if not user_question:
        st.caption("Ketikkan pertanyaan untuk mencoba simulasi proses!")

    # Proses embeddings pertanyaan
    with st.expander("Proses Embeddings Pertanyaan", expanded=bool(user_question)):
        if user_question:
            st.write(f"Pertanyaan pengguna: {user_question}")
            progress = st.progress(0)
            user_embedding = EMBEDDER.embed_query(user_question)
            progress.progress(100)
            st.write("Embeddings pertanyaan berhasil.")
            df_embedding = pd.DataFrame([{"Pertanyaan": user_question, "Embedding": user_embedding}])
            st.dataframe(df_embedding, use_container_width=True)
        else:
            st.warning("Menunggu pertanyaan dari pengguna...", icon="⏳")

    # Proses similarity search menggunakan FAISS
    with st.expander("Proses Similarity Search", expanded=bool(user_question)):
        if user_question:
            st.write("Sedang memproses similarity search...")
            progress = st.progress(0)
            vectordb = FAISS.load_local("src/vectordb", EMBEDDER, allow_dangerous_deserialization=True)
            retriever = vectordb.similarity_search_with_relevance_scores(user_question, k=5)
            progress.progress(100)
            if retriever and len(retriever) > 0:
                candidate_data = [{"No": i + 1, "Isi Chunk (Top 5)": doc.page_content, "Similarity Score": score} for i, (doc, score) in enumerate(retriever)]
                df_candidates = pd.DataFrame(candidate_data)
                st.table(df_candidates)
            else:
                st.write("Tidak ada hasil yang ditemukan.")
        else:
            st.warning("Menunggu pertanyaan dari pengguna...", icon="⏳")

    # Proses embeddings kandidat
    with st.expander("Proses Embeddings Kandidat", expanded=bool(user_question)):
        if user_question:
            if retriever and len(retriever) > 0:
                st.write("Sedang memproses embeddings kandidat chunk...")
                progress = st.progress(0)
                candidate_embeddings = [EMBEDDER.embed_query(doc.page_content) for doc, _ in retriever]
                progress.progress(100)
                candidate_embedding_data = [{"No": i + 1, "Isi Chunk (Top 5)": doc.page_content, "Embedding": emb} for i, (doc, emb) in enumerate(zip([doc for doc, _ in retriever], candidate_embeddings))]
                df_candidate_embeddings = pd.DataFrame(candidate_embedding_data)
                st.dataframe(df_candidate_embeddings, use_container_width=True)
            else:
                st.write("Tidak ada kandidat chunk untuk di-embedding.")
        else:
            st.warning("Menunggu pertanyaan dari pengguna...", icon="⏳")

    # Menyiapkan hasil jawaban akhir
    with st.expander("Proses Menyiapkan Hasil", expanded=bool(user_question)):
        if user_question:
            if retriever and len(retriever) > 0:
                prompt = f"""
                    Pertanyaan: {user_question}
                    Konteks: {retriever}
                """
                messages = [
                    HumanMessage(prompt)
                ]
                response = chat_openai(messages)
                st.write(f"Pertanyaan: \n{user_question}")
                st.write(f"Jawaban: \n{response}")
            else:
                st.write("Tidak ada hasil yang ditemukan.")
        else:
            st.warning("Menunggu pertanyaan dari pengguna...", icon="⏳")


def debug_key():
    if "access_granted" not in st.session_state:
        st.session_state.access_granted = False
    
    with st.sidebar.popover("Login Admin"):
        input_placeholder = st.empty()
        user_key = input_placeholder.text_input("Masukkan Key Admin untuk debugging!", type="password")

    if user_key:
        if user_key == STREAMLIT_KEY_ADMIN:
            st.session_state.access_granted = True
            input_placeholder.success("Key Admin valid. Berhasil masuk ke debugging!")
        else:
            st.session_state.access_granted = False
            st.warning("Key Admin tidak valid. Silakan coba lagi!")

    if st.session_state.access_granted:
        entryDatasets()
        st.markdown("***")
        raw_process()
        st.markdown("***")
        query_user_process()


def main():
    setup_page()
    debug_key()


if __name__ == "__main__":
    main()