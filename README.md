# VIRTUAL ASSISTANT PMB UNDIKSHA

## Ringkasan

![image](assets/images/architecture.jpg)
Proyek ini mengembangkan Virtual Assistant (VA) Berbasis Multi-Agent LLM dengan Teknik Adaptive-RAG pada Sistem Penerimaan Mahasiswa Baru (PMB) di Universitas Pendidikan Ganesha (Undiksha). Menggunakan OpenAI sebagai LLM, LangChain untuk proses RAG, Langgraph untuk mengelola Multi-Agent LLM Adaptive-RAG, dan FAISS sebagai vector database. Virtual Assistant ini dirancang untuk memberikan informasi yang akurat dan cepat, meningkatkan efisiensi dan pengalaman pengguna dalam proses penerimaan mahasiswa.

## Permasalahan

Di periode-periode sebelumnya, pada Sistem Informasi PMB Undiksha masih mengandalkan penyebaran informasi secara manual melalui website atau grup sosial media seperti Telegram, yang mengharuskan pengguna untuk secara aktif mencari informasi. Pendekatan ini memiliki beberapa kelemahan, yaitu:

- Tidak efisien dan memakan waktu lebih lama.
- Memerlukan interaksi langsung dengan admin atau sistem yang tidak fleksibel.
- Seringkali calon mahasiswa mengulang pertanyaan yang sebenarnya sudah terjawab.
  Hal ini membuat pengalaman pengguna kurang optimal dan menyulitkan dalam menghadapi volume pertanyaan yang terus berkembang.

## Solusi

Virtual Assistant PMB Undiksha menawarkan solusi dengan teknologi terkini untuk mengatasi permasalahan di atas. Dengan memanfaatkan Retrieval Augmented Generation (RAG), Virtual Assistant ini dapat:

- Meningkatkan efisiensi dengan jawaban relevan dan akurat dari database dengan instan tanpa menunggu balasan langsung dari admin.
- Menjawab pertanyaan sering diajukan dengan cepat dan konsisten.
- Memungkinkan pembaruan informasi mudah dan responsif terhadap kebutuhan terbaru.

## Teknologi

- [Python](https://www.python.org/): Bahasa pemrograman untuk membuat Chatbot.
- [Langchain](https://www.langchain.com/): Framework untuk mengelola alur kerja RAG.
- [Langgraph](https://www.langchain.com/langgraph): Framework untuk Multi Agent pada Langchain.
- [OpenAI](https://openai.com/): Embedding dan model RAG berbayar.
- [Streamlit](https://streamlit.io/): Web interface application.
- [FAISS](https://faiss.ai/): Penyimpanan vector database.

## Apa itu RAG?

![image](assets/images/rag.png)
Retrieval-Augmented Generation (RAG) adalah teknik yang dirancang untuk meningkatkan kinerja Large Language Model (LLM) dengan mengakses informasi dari sumber eksternal. Dengan RAG, Virtual Assistant dapat memberikan jawaban yang lebih akurat dan relevan, serta mengurangi kemungkinan halusinasi terhadap suatu informasi.

## Alur Kerja RAG

#### 1. Retrieve (Kumpulkan):

- Kueri dari pengguna digunakan untuk mencari konteks relevan dari sumber pengetahuan eksternal.
- Kuery diubah menjadi vektor dan dicocokkan dengan vektor dalam database (sumber pengetahuan juga telah melewati fase ini), sehingga mendapatkan objek data relevan (k untuk objek paling relevan).

#### 2. Augment (Tambahkan):

- Konteks diambil dan digabungkan dengan kueri pengguna menggunakan template prompt.

#### 3. Generate (Hasilkan Respon):

- Prompt yang sudah dimodifikasi struktur datanya dimasukkan ke dalam LLM untuk menghasilkan respons akhir.

## Contoh Implementasi

![image](assets/images/concept.jpg)
Pertanyaan Pengguna (Kueri) "Apa syarat untuk mendaftar sebagai mahasiswa baru di Undiksha?"

#### 1. Retrieve

Konteks relevan diambil dari database vektor.
Konteks: "Untuk mendaftar sebagai mahasiswa baru di Undiksha, calon mahasiswa harus memiliki ijazah SMA atau sederajat, melengkapi formulir pendaftaran, dan mengikuti ujian masuk."

#### 2. Augment

Gabungkan kuery pengguna dengan konteks yang diambil menggunakan template prompt.
Prompt: "Syarat-syarat pendaftaran mahasiswa baru di UndikshaSyarat-syarat pendaftaran mahasiswa baru di Undiksha adalah sebagai berikut: sebagai berikut: {context}"

#### 3. Generate

LLM memproses prompt tersebut untuk menghasilkan respons lengkap.
Respons Akhir: "Syarat-syarat pendaftaran mahasiswa baru di Undiksha adalah sebagai berikut: Untuk mendaftar sebagai mahasiswa baru di Undiksha, calon mahasiswa harus memiliki ijazah SMA atau sederajat, melengkapi formulir pendaftaran, dan mengikuti ujian masuk."

## Instalasi Project

Clone project

```bash
  https://github.com/odetv/va-pmb-undiksha.git
```

Masuk ke direktori project

```bash
  cd va-pmb-undiksha
```

Install Requirements

```bash
  pip install -r requirements.txt
```

Buat dan Lengkapi file environment variabel (.env)

```bash
  OPENAI_API_KEY=""
  GROQ_API_KEY=""
  OLLAMA_BASE_URL=""
  API_KTM_UNDIKSHA_AUTH_URL=""
  API_KTM_UNDIKSHA_USERNAME=""
  API_KTM_UNDIKSHA_PASSWORD=""
  API_KTM_UNDIKSHA_RESPONSE_URL=""
  API_KELULUSAN_UNDIKSHA_AUTH_URL=""
  API_KELULUSAN_UNDIKSHA_USERNAME=""
  API_KELULUSAN_UNDIKSHA_PASSWORD=""
  API_KELULUSAN_UNDIKSHA_RESPONSE_URL=""
  STREAMLIT_KEY_ADMIN=""
```

Jalankan project

```bash
  streamlit run app/Home.py
```

## Referensi

1. [Build a ChatBot Using Local LLM](https://datasciencenerd.us/build-a-chatbot-using-local-llm-6b8dbb0ca514)
2. [Best Practices in Retrieval Augmented Generation](https://gradientflow.substack.com/p/best-practices-in-retrieval-augmented)
3. [Simplest Method to improve RAG pipeline: Re-Ranking](https://medium.com/etoai/simplest-method-to-improve-rag-pipeline-re-ranking-cf6eaec6d544)
4. [The What and How of RAG(Retrieval Augmented Generation) Implementation Using Langchain](https://srinivas-mahakud.medium.com/the-what-and-how-of-retrieval-augmented-generation-8e4a05c08a50)
5. [Retrieval-Augmented Generation (RAG): From Theory to LangChain Implementation](https://towardsdatascience.com/retrieval-augmented-generation-rag-from-theory-to-langchain-implementation-4e9bd5f6a4f2)
6. [RAG - PDF Q&A Using Llama 2 in 8 Steps](https://medium.com/@Sanjjushri/rag-pdf-q-a-using-llama-2-in-8-steps-021a7dbe26e1)
7. [RAG + Langchain Python Project: Easy AI/Chat For Your Docs](https://youtu.be/tcqEUSNCn8I)
8. [Python RAG Tutorial (with Local LLMs): Al For Your PDFs](https://youtu.be/2TJxpyO3ei4)
9. [A Survey of Techniques for Maximizing LLM Performance](https://youtu.be/ahnGLM-RC1Y)
10. [18 Lessons teaching everything you need to know to start building Generative AI applications](https://microsoft.github.io/generative-ai-for-beginners/#/)
11. [How to build a PDF chatbot with Langchain 🦜🔗 and FAISS](https://kevincoder.co.za/how-to-build-a-pdf-chatbot-with-langchain-and-faiss)
12. [How to build a PDF chatbot with Langchain 🦜🔗 and FAISS](https://heartbeat.comet.ml/how-to-enhance-conversational-agents-with-memory-in-lang-chain-6aadd335b621)
13. [Memory in LLMChain](https://python.langchain.com/v0.1/docs/modules/memory/adding_memory/)
14. [RunnableWithMessageHistory](https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.history.RunnableWithMessageHistory.html#langchain_core.runnables.history.RunnableWithMessageHistory)
15. [Why Assistants API is Slow? Any speed solution?](https://community.openai.com/t/why-assistants-api-is-slow-any-speed-solution/558065)
16. [OpenAI API is extremely slow](https://github.com/langchain-ai/langchain/issues/11836)
17. [Adaptive RAG](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag/)
18. [Hands-On LangChain for LLMs App: ChatBots Memory](https://pub.towardsai.net/hands-on-langchain-for-llms-app-chatbots-memory-9394030e5a9e)
19. [How to Make LLM Remember Conversation with Langchain](https://medium.com/@vinayakdeshpande111/how-to-make-llm-remember-conversation-with-langchain-924083079d95)
20. [Conversation Summary Buffer](https://python.langchain.com/v0.1/docs/modules/memory/types/summary_buffer/)
21. [From Basics to Advanced: Exploring LangGraph](https://towardsdatascience.com/from-basics-to-advanced-exploring-langgraph-e8c1cf4db787)
22. [Build a Reliable RAG Agent using LangGraph](https://medium.com/the-ai-forum/build-a-reliable-rag-agent-using-langgraph-2694d55995cd)
23. [LangGraph](https://langchain-ai.github.io/langgraph/)
