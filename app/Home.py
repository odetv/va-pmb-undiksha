import streamlit as st
import re
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import build_graph


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


def main():
    title_desc()

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