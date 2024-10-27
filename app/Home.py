import streamlit as st
import re
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import build_graph


EXAMPLE_QUESTIONS = [
    "Bagaimana cara mendaftar di Undiksha? Ajarin saya dong.",
    "Jalur masuk apa saja yang tersedia saat mendaftar di Undiksha?",
    "Dimana lokasi kampus Undiksha? Katanya ada juga di Denpasar.",
    "Saya ingin cek kelulusan pendaftaran di Undiksha.",
    "Bagaimana cara akses melihat Kartu Tanda Mahasiswa?"
]
INITIAL_MESSAGE = {"role": "assistant", "content": "Salam Harmoni🙏 Ada yang bisa saya bantu?", "raw_content": "Salam Harmoni🙏 Ada yang bisa saya bantu?", "images": []}


def setup_page():
    st.set_page_config(page_title="VA PMB Undiksha", layout="wide", page_icon="assets/images/logo.png")
    st.sidebar.image("assets/images/logo.png", use_column_width=True)
    st.sidebar.title("Virtual Assistant PMB Undiksha")
    st.sidebar.write("Hai, selamat datang di Virtual Assistant Penerimaan Mahasiswa Baru Undiksha! Aku siap membantumu.")
    st.sidebar.markdown("""
    <p style="color:gray;">
        <small>Developed by: <strong>I Gede Gelgel Abdiutama</strong></small><br>
        <small>Support by: <strong>UPA TIK Undiksha</strong></small>
    </p>
    """, unsafe_allow_html=True)
    st.title("Ayo tanyakan padaku😊")


def process_response(prompt):
    with st.spinner("Sedang memproses, harap tunggu..."):
        response = build_graph(prompt)
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
    cols = st.columns(len(EXAMPLE_QUESTIONS), vertical_alignment="center")
    for col, prompt in zip(cols, EXAMPLE_QUESTIONS):
        with col:
            if st.button(prompt):
                add_message("user", prompt)
                st.session_state['user_question'] = prompt
                response = process_response(prompt)
                add_message("assistant", response["msg"], response["html_msg"], response["images"])
                st.rerun()


def display_chat_history():
    for msg in st.session_state.messages:
        st.chat_message(msg["role"], avatar="assets/images/avatar-va.png").write(msg["raw_content"])
        for img_url in msg.get("images", []):
            st.image(img_url, use_column_width=False)


def handle_user_input():
    if prompt := st.chat_input("Ketik pertanyaan Anda di sini..."):
        add_message("user", prompt)
        st.session_state['user_question'] = prompt
        st.chat_message("user", avatar="assets/images/avatar-user.png").write(prompt)
        response = process_response(prompt)
        add_message("assistant", response["msg"], response["html_msg"], response["images"])
        st.chat_message("assistant", avatar="assets/images/avatar-va.png").markdown(response["msg"])
        for img_url in response["images"]:
            st.image(img_url, use_column_width=False)


def main():
    setup_page()
    display_example_questions()
    st.markdown("***")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [INITIAL_MESSAGE]
    
    display_chat_history()
    handle_user_input()

if __name__ == "__main__":
    main()