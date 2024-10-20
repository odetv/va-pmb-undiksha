import streamlit as st
import re
from main import build_graph


def show_example_questions():
    example_questions = [
        "Apa persyaratan PMB?",
        "Bagaimana cara mendaftar?",
        "Apa saja jalur masuk yang tersedia?",
        "Kapan jadwal tes masuk?"
    ]
    
    cols = st.columns(4)
    
    for i, question in enumerate(example_questions):
        with cols[i]:
            if st.button(question):
                st.session_state.user_question = question

def main():
    st.title("Virtual Assistant PMB Undiksha")
    st.write("Salam Harmoni🙏 Silahkan bertanya😊")

    show_example_questions()

    user_question = st.text_input("Pertanyaan:")

    if st.button("Kirim"):
        if user_question:
            with st.spinner("Sedang memproses, harap tunggu..."):
                response = build_graph(user_question)
            st.success("Pertanyaan telah diproses.")

            response_with_links = re.sub(
                r'(https://aka\.undiksha\.ac\.id/api/ktm/generate/\S*)', 
                r'<a href="\1" target="_blank">[Preview URL]</a>', 
                response
            )
            st.markdown(response_with_links, unsafe_allow_html=True)

            image_links = re.findall(r'(https?://\S+)', response)
            for link in image_links:
                if "https://aka.undiksha.ac.id/api/ktm/generate" in link:
                    st.image(link, use_column_width=True)
                elif link.endswith(('.jpg', '.jpeg', '.png', '.gif')):
                    st.image(link, use_column_width=True)

        else:
            st.warning("Tolong masukkan pertanyaan Anda.")

if __name__ == "__main__":
    main()
