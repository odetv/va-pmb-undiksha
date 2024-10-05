# CHATBOT INFO PMB UNDIKSHA (LANGCHAIN)

## Ringkasan

Proyek ini mengembangkan chatbot berbasis Retrieval Augmented Generation (RAG) untuk Sistem Informasi Penerimaan Mahasiswa Baru (PMB) di Universitas Pendidikan Ganesha (Undiksha). Menggunakan Ollama, LangChain, ChromaDB, dan FastAPI, chatbot ini dirancang untuk memberikan informasi yang akurat dan cepat, meningkatkan efisiensi dan pengalaman pengguna dalam proses penerimaan mahasiswa.

## Permasalahan

Pada periode-periode sebelumnya, Sistem Informasi PMB Undiksha masih mengandalkan penyebaran informasi secara manual melalui website atau grup sosial media seperti Telegram, yang mengharuskan pengguna untuk secara aktif mencari informasi. Pendekatan ini memiliki beberapa kelemahan, yaitu:

- Tidak efisien dan memakan waktu.
- Memerlukan interaksi langsung dengan admin atau sistem yang tidak fleksibel.
- Seringkali calon mahasiswa mengulang pertanyaan yang sebenarnya sudah terjawab.
  Hal ini membuat pengalaman pengguna kurang optimal dan menyulitkan dalam menghadapi volume pertanyaan yang terus berkembang.

## Solusi

Chatbot Info PMB Undiksha menawarkan solusi modern untuk mengatasi permasalahan di atas. Dengan memanfaatkan teknologi Retrieval Augmented Generation (RAG), chatbot ini dapat:

- Meningkatkan efisiensi dengan jawaban relevan dan akurat dari database dengan instan tanpa menunggu balasan langsung dari admin.
- Menjawab pertanyaan sering diajukan dengan cepat dan konsisten.
- Memungkinkan pembaruan informasi mudah dan responsif terhadap kebutuhan terbaru.

## Teknologi

- [Python](https://www.python.org/): Bahasa pemrograman untuk membuat Chatbot.
- [Ollama](https://ollama.com/): Embedding dan model RAG Lokal Gratis (bge-m3 & llama3.1).
- [Langchain](https://www.langchain.com/): Framework untuk mengelola alur kerja RAG.
- [ChromaDB](https://www.trychroma.com/): Penyimpanan database vektor.
- [FastAPI](https://fastapi.tiangolo.com/): Framework backend untuk membuat API.

## Apa itu RAG?

![image](https://gradientflow.com/wp-content/uploads/2023/10/newsletter87-RAG-simple.png)
Retrieval-Augmented Generation (RAG) adalah teknik yang dirancang untuk meningkatkan kinerja Large Language Model (LLM) dengan mengakses informasi dari sumber eksternal. Dengan RAG, chatbot dapat memberikan jawaban yang lebih akurat dan relevan, serta mengurangi kemungkinan halusinasi terhadap suatu informasi.

## Alur Kerja RAG

#### 1. Retrieve (Kumpulkan):

- Kueri dari pengguna digunakan untuk mencari konteks relevan dari sumber pengetahuan eksternal.
- Kuery diubah menjadi vektor dan dicocokkan dengan vektor dalam database (sumber pengetahuan juga telah melewati fase ini), sehingga mendapatkan objek data relevan (k untuk objek paling relevan).

#### 2. Augment (Tambahkan):

- Konteks diambil dan digabungkan dengan kueri pengguna menggunakan template prompt.

#### 3. Generate (Hasilkan Respon):

- Prompt yang sudah dimodifikasi struktur datanya dimasukkan ke dalam LLM untuk menghasilkan respons akhir.

## Contoh Implementasi

![image](https://miro.medium.com/v2/resize:fit:828/format:webp/1*h5SO9Hqu1YVYQVEIsWGZBg.png)
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
  https://github.com/odetv/chatbot-pmb-undiksha.git
```

Masuk ke direktori project

```bash
  cd chatbot-pmb-undiksha
```

Install Requirements

```bash
  pip install -r requirements.txt
```

Buat dan Lengkapi file environment variabel (.env)

```bash
  OPENAI_API_KEY="API_KEY_OPENAI_PLATFORM"
  CHATBOT_API_KEY="API_KEY_KEAMANAN_APLIKASI_BEBAS"
```

Jalankan project

```bash
  uvicorn main:app --host=0.0.0.0 --port=8000 --reload
```

## Referensi

- [Best Practices in Retrieval Augmented Generation](https://gradientflow.substack.com/p/best-practices-in-retrieval-augmented)
- [Simplest Method to improve RAG pipeline: Re-Ranking](https://medium.com/etoai/simplest-method-to-improve-rag-pipeline-re-ranking-cf6eaec6d544)
- [The What and How of RAG(Retrieval Augmented Generation) Implementation Using Langchain](https://srinivas-mahakud.medium.com/the-what-and-how-of-retrieval-augmented-generation-8e4a05c08a50)
- [Retrieval-Augmented Generation (RAG): From Theory to LangChain Implementation](https://towardsdatascience.com/retrieval-augmented-generation-rag-from-theory-to-langchain-implementation-4e9bd5f6a4f2)
- [RAG - PDF Q&A Using Llama 2 in 8 Steps](https://medium.com/@Sanjjushri/rag-pdf-q-a-using-llama-2-in-8-steps-021a7dbe26e1)
- [RAG + Langchain Python Project: Easy AI/Chat For Your Docs](https://youtu.be/tcqEUSNCn8I)
- [Python RAG Tutorial (with Local LLMs): Al For Your PDFs](https://youtu.be/2TJxpyO3ei4)
- [A Survey of Techniques for Maximizing LLM Performance](https://youtu.be/ahnGLM-RC1Y)
- [18 Lessons teaching everything you need to know to start building Generative AI applications](https://microsoft.github.io/generative-ai-for-beginners/#/)
- [How to build a PDF chatbot with Langchain 🦜🔗 and FAISS](https://kevincoder.co.za/how-to-build-a-pdf-chatbot-with-langchain-and-faiss)
- [Memory in LLMChain](https://python.langchain.com/v0.1/docs/modules/memory/adding_memory/)
- [How to Make LLM Remember Conversation with Langchain](https://medium.com/@vinayakdeshpande111/how-to-make-llm-remember-conversation-with-langchain-924083079d95)
- [Conversation Summary Buffer](https://python.langchain.com/v0.1/docs/modules/memory/types/summary_buffer/)
