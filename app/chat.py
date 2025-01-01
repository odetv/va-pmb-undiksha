import streamlit as st
import re
import sys
import os
import firebase_admin
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from firebase_admin import credentials, firestore
from main import rag_adaptive


# Tokenization
from dotenv import load_dotenv
load_dotenv()
STREAMLIT_KEY_ADMIN = os.getenv("STREAMLIT_KEY_ADMIN")
st.set_page_config(page_title="VA PMB Undiksha", layout="wide", page_icon="public/images/logo.png")
@st.cache_resource
def init_firebase():
    cred = credentials.Certificate({
        "type": os.getenv("FIREBASE_TYPE"),
        "project_id": os.getenv("FIREBASE_PROJECT_ID"),
        "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
        "private_key": os.getenv("FIREBASE_PRIVATE_KEY").replace("\\n", "\n"), # Berhati-hati dengan newlines di private key
        "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
        "client_id": os.getenv("FIREBASE_CLIENT_ID"),
        "auth_uri": os.getenv("FIREBASE_AUTH_URI"),
        "token_uri": os.getenv("FIREBASE_TOKEN_URI"),
        "auth_provider_x509_cert_url": os.getenv("FIREBASE_AUTH_PROVIDER_CERT_URL"),
        "client_x509_cert_url": os.getenv("FIREBASE_CLIENT_CERT_URL"),
        "universe_domain": os.getenv("FIREBASE_UNIVERSE_DOMAIN")
    })
    try:
        firebase_admin.get_app()
    except ValueError:
        firebase_admin.initialize_app(cred)
    return firestore.client()
db = init_firebase()
def verify_token(user_token):
    tokens_ref = db.collection('tokens')
    docs = tokens_ref.where('token', '==', user_token).where('status', '==', 1).stream()
    for doc in docs:
        return doc.exists
    return False
def get_auth_status():
    settings_ref = db.collection("settings").document("auth")
    doc = settings_ref.get()
    if doc.exists:
        return doc.to_dict().get("status", 1)
    return 0


EXAMPLE_QUESTIONS = [
    "Bagaimana cara mendaftar di Undiksha? Ajarin saya dong.",
    "Jalur masuk apa saja yang tersedia saat mendaftar di Undiksha?",
    "Dimana lokasi kampus Undiksha? Katanya ada juga di Denpasar.",
    "Saya ingin cek kelulusan pendaftaran di Undiksha.",
    "Bagaimana cara akses melihat Kartu Tanda Mahasiswa?"
]
INITIAL_MESSAGE = {"role": "assistant", "content": "Salam Harmoni🙏 Ada yang bisa dibantu?", "raw_content": "Salam Harmoni🙏 Ada yang bisa dibantu?", "images": []}


def setup_page():
    # st.set_page_config(page_title="VA PMB Undiksha", layout="wide", page_icon="public/images/logo.png")
    st.sidebar.image("public/images/logo.png")
    st.sidebar.title("Virtual Assistant PMB Undiksha")
    st.sidebar.write("Hai Ganesha Muda, selamat datang di Virtual Assistant Penerimaan Mahasiswa Baru Undiksha! Aku siap membantumu.")
    st.sidebar.markdown("""
    <p style="color:gray;">
        <small>Author: <strong>I Gede Gelgel Abdiutama</strong></small><br>
        <small>Support: <strong>UPA TIK Undiksha</strong></small>
    </p>
    """, unsafe_allow_html=True)
    st.title("Tanya Ganesha Muda🎓")


def process_response(question):
    with st.spinner("Sedang memproses, harap tunggu..."):
        _, response = rag_adaptive(question)
        msg = re.sub(
            r'(https://aka\.undiksha\.ac\.id/api/ktm/generate/\S*)', 
            r'[Preview URL](\1)',
            response
        )
        html_msg = re.sub(
            r'(https://aka\.undiksha\.ac\.id/api/ktm/generate/\S*)', 
            r'<a href="\1" target="_blank">[Preview URL]</a>', 
            response
        )
        images = [
            link for link in re.findall(r'(https?://\S+)', msg)
            if "https://aka.undiksha.ac.id/api/ktm/generate" in link or link.endswith(('.jpg', '.jpeg', '.png', '.gif'))
        ]
        return {"msg": msg, "html_msg": html_msg, "images": images}


def add_message(role, content, html_content=None, images=None):
    if "messages" not in st.session_state:
        st.session_state.messages = [INITIAL_MESSAGE]
    message = {
        "role": role,
        "content": html_content if html_content else content,
        "raw_content": content,
        "images": images or []
    }
    st.session_state.messages.append(message)


def display_example_questions():
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for col, question in zip(cols, EXAMPLE_QUESTIONS):
        with col:
            if st.button(question):
                add_message("user", question)
                st.session_state['user_question'] = question
                response = process_response(question)
                add_message("assistant", response["msg"], response["html_msg"], response["images"])
                st.rerun()


def display_chat_history():
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["raw_content"])
        for img_url in msg.get("images", []):
            st.image(img_url)


def handle_user_input():
    if question := st.chat_input("Ketik pertanyaan Anda di sini"):
        add_message("user", question)
        st.session_state['user_question'] = question
        st.chat_message("user").write(question)
        response = process_response(question)
        add_message("assistant", response["msg"], response["html_msg"], response["images"])
        st.chat_message("assistant").markdown(response["msg"])
        for img_url in response["images"]:
            st.image(img_url)


def main():
    setup_page()
    auth_status = get_auth_status()

    if auth_status == 1:
        # Tokenization
        if "authenticated" not in st.session_state:
            st.session_state.authenticated = False
        if not st.session_state.authenticated:
            placeholder = st.empty()
            with placeholder.container():
                user_token = st.text_input("Masukkan token untuk dapat mengakses Virtual Assistant", type="password")
                if st.button("Submit"):
                    if verify_token(user_token) or user_token == STREAMLIT_KEY_ADMIN:
                        st.session_state.authenticated = True
                        st.success("Token valid. Selamat datang!")
                        placeholder.empty()
                    else:
                        st.error("Token salah, coba lagi!")
        if st.session_state.authenticated:
            display_example_questions()
            st.markdown("***")
            if "messages" not in st.session_state:
                st.session_state.messages = [INITIAL_MESSAGE]
            display_chat_history()
            handle_user_input()
    else:
        # No tokenization
        display_example_questions()
        st.markdown("***")
        if "messages" not in st.session_state:
            st.session_state.messages = [INITIAL_MESSAGE]
        display_chat_history()
        handle_user_input()

if __name__ == "__main__":
    main()