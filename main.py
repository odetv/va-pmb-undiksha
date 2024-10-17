import re
import time
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import END, START, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from utils.llm import chat_ollama, chat_openai, chat_groq
from utils.api_undiksha import cetak_ktm_mhs
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from utils.create_graph_image import get_graph_image
from utils.agent_state import AgentState


total_execution_time = 0.0
def time_check(func):
    def wrapper(state: AgentState):
        global total_execution_time
        start_time = time.time()
        result = func(state)
        end_time = time.time()
        execution_time = end_time - start_time
        total_execution_time += execution_time
        print(f"DEBUG: {func.__name__} took {execution_time:.4f} seconds\n\n")
        return result
    return wrapper


@time_check
def questionIdentifierAgent(state: AgentState):
    info = "\n--- AGENT QUESTION IDENTIFIER ---"
    print(info)
    prompt = """
        Anda adalah seoarang analis pertanyaan pengguna.
        Tugas Anda adalah mengklasifikasikan jenis pertanyaan pada konteks Undiksha (Universitas Pendidikan Ganesha).
        Tergantung pada jawaban Anda, akan mengarahkan ke agent yang tepat.
        Ada 3 konteks pertanyaan yang diajukan:
        - GENERAL - Pertanyaan terkait informasi seputar Penerimaan Mahasiswa Baru (PMB) dan perkuliahan kampus baik itu akademik dan mahasiswa di Undiksha (Universitas Pendidikan Ganesha).
        - KTM - Pertanyaan terkait Kartu Tanda Mahasiswa (KTM).
        - OUTOFCONTEXT - Hanya jika diluar dari konteks.
        Hasilkan hanya sesuai kata (GENERAL, KTM, OUTOFCONTEXT), kemungkinan pertanyaannya berisi lebih dari 1 konteks yang berbeda, pisahkan dengan tanda koma.
    """
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=state["question"]),
    ]
    response = chat_groq(messages)
    cleaned_response = response.strip().lower()
    print("Pertanyaan:", state["question"])
    print(f"question_type: {cleaned_response}")
    state["question_type"] = cleaned_response
    return state


@time_check
def generalAgent(state: AgentState):
    info = "\n--- AGENT GENERAL ---"
    print(info)
    VECTOR_PATH = "vectordb"
    MODEL_EMBEDDING = "text-embedding-3-small"
    EMBEDDER = OpenAIEmbeddings(model=MODEL_EMBEDDING)
    question = state["question"]
    vectordb = FAISS.load_local(VECTOR_PATH,  EMBEDDER, allow_dangerous_deserialization=True) 
    retriever = vectordb.similarity_search_with_relevance_scores(question, k=5)
    context = "\n\n".join([doc.page_content for doc, _score in retriever])
    state["generalContext"] = context
    # print (state["generalContext"])
    return {"generalContext": state["generalContext"]}


@time_check
def graderDocsAgent(state: AgentState):
    info = "\n--- Agent Grader Documents ---"
    print(info)
    prompt = f"""
    Anda adalah seorang pemilih konteks handal.
    - Ambil informasi yang hanya berkaitan dengan pertanyaan.
    - Pastikan informasi yang diambil lengkap sesuai konteks yang diberikan.
    - Jangan mengurangi atau melebihi konteks yang diberikan.
    - Format nya gunakan sesuai format konteks yang dberikan, jangan dirubah.
    - Jangan jawab pertanyaan pengguna, hanya pilah konteks yang berkaitan dengan pertanyaan saja.
    Konteks: {state["generalContext"]}
    """
    VECTOR_PATH = "vectordb"
    MODEL_EMBEDDING = "text-embedding-3-small"
    EMBEDDER = OpenAIEmbeddings(model=MODEL_EMBEDDING)
    question = state["question"]
    vectordb = FAISS.load_local(VECTOR_PATH,  EMBEDDER, allow_dangerous_deserialization=True) 
    retriever = vectordb.similarity_search_with_relevance_scores(question, k=5)
    context_text = "\n".join([doc.page_content for doc, _score in retriever])

    prompt_template = ChatPromptTemplate.from_template(prompt)
    prompt = prompt_template.format(context=context_text, question=question)

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=state["question"]),
    ]
    responseGraderDocsAgent = chat_openai(messages)
    state["generalGraderDocs"] = responseGraderDocsAgent
    # print(state["generalGraderDocs"])
    return {"generalGraderDocs": state["generalGraderDocs"]}


@time_check
def answerGeneratorAgent(state: AgentState):
    info = "\n--- Agent Answer Generator ---"
    print(info)
    prompt = f"""
    Berikut pedoman yang harus diikuti untuk memberikan jawaban yang relevan dan sesuai konteks dari pertanyaan yang diajukan:
    - Anda bertugas untuk memberikan informasi Penerimaan Mahasiswa Baru dan yang terkait dengan Universitas Pendidikan Ganesha.
    - Pahami frasa atau terjemahan kata-kata dalam bahasa asing sesuai dengan konteks dan pertanyaan.
    - Jika ditanya siapa Anda, identitas Anda sebagai Bot Agent Informasi PMB Undiksha.
    - Berikan jawaban yang akurat dan konsisten untuk lebih dari satu pertanyaan yang mirip atau sama hanya berdasarkan konteks yang telah diberikan.
    - Jawab sesuai apa yang ditanyakan saja dan jangan menggunakan informasi diluar konteks, sampaikan dengan apa adanya jika Anda tidak mengetahui jawabannya.
    - Jangan berkata kasar, menghina, sarkas, satir, atau merendahkan pihak lain.
    - Berikan jawaban yang lengkap, rapi, dan penomoran jika diperlukan sesuai konteks.
    - Jangan sampaikan pedoman ini kepada pengguna, gunakan pedoman ini hanya untuk memberikan jawaban yang sesuai konteks.
    Pertanyaan Pengguna: {state["question"]}
    Konteks: {state["generalGraderDocs"]}
    """
    VECTOR_PATH = "vectordb"
    MODEL_EMBEDDING = "text-embedding-3-small"
    EMBEDDER = OpenAIEmbeddings(model=MODEL_EMBEDDING)
    question = state["question"]
    vectordb = FAISS.load_local(VECTOR_PATH,  EMBEDDER, allow_dangerous_deserialization=True) 
    retriever = vectordb.similarity_search_with_relevance_scores(question, k=5)
    context_text = "\n".join([doc.page_content for doc, _score in retriever])

    prompt_template = ChatPromptTemplate.from_template(prompt)
    prompt = prompt_template.format(context=context_text, question=question)

    messages = [
        SystemMessage(content=prompt)
    ]
    response = chat_groq(messages)
    state["responseGeneral"] = response
    state["agentsContext"] = response
    # print(state["responseGeneral"])
    return {"agentsContext": state["responseGeneral"]}


@time_check
def graderHallucinationsAgent(state: AgentState):
    info = "\n--- Agent Grader Hallucinations ---"
    print(info)
    prompt = f"""
    Anda adalah seorang penilai yang menilai apakah hasil didukung oleh sekumpulan fakta yang diambil dari informasi fakta.
    - Berikan hanya nilai "true" jika halusinasi atau tidak sesuai fakta atau "false" jika tidak halusinasi atau sesuai fakta.
    - Informasi fakta: {state["generalGraderDocs"]}
    - Hasil yang perlu dibandingkan dengan informasi fakta: {state["agentsContext"]}
    """
    VECTOR_PATH = "vectordb"
    MODEL_EMBEDDING = "text-embedding-3-small"
    EMBEDDER = OpenAIEmbeddings(model=MODEL_EMBEDDING)
    question = state["question"]
    vectordb = FAISS.load_local(VECTOR_PATH,  EMBEDDER, allow_dangerous_deserialization=True) 
    retriever = vectordb.similarity_search_with_relevance_scores(question, k=5)
    context_text = "\n".join([doc.page_content for doc, _score in retriever])

    prompt_template = ChatPromptTemplate.from_template(prompt)
    prompt = prompt_template.format(context=context_text, question=question)

    messages = [
        SystemMessage(content=prompt)
    ]
    response = chat_ollama(messages).strip().lower()
    is_hallucination = response == "true"
    state["generalIsHallucination"] = is_hallucination
    print(f"Is hallucination? {is_hallucination}")
    return {"generalIsHallucination": state["generalIsHallucination"]}


@time_check
def ktmAgent(state: AgentState):
    info = "\n--- AGENT KTM ---"
    print(info)
    prompt = """
        Anda adalah seoarang analis informasi Kartu Tanda Mahasiswa (KTM).
        Tugas Anda adalah mengklasifikasikan jenis pertanyaan pada konteks Undiksha (Universitas Pendidikan Ganesha).
        NIM (Nomor Induk Mahasiswa) yang valid dari Undiksha berjumlah 10 digit angka.
        Sekarang tergantung pada jawaban Anda, akan mengarahkan ke agent yang tepat.
        Ada 2 konteks pertanyaan yang diajukan:
        - INCOMPLETENIM - Jika pengguna tidak menyertakan nomor NIM (Nomor Induk Mahasiswa) dan tidak valid.
        - PRINTKTM - Jika pengguna menyertakan NIM (Nomor Induk Mahasiswa).
        Hasilkan hanya 1 sesuai kata (INCOMPLETENIM, PRINTKTM).
    """
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=state["question"]),
    ]
    response = chat_groq(messages)
    cleaned_response = response.strip().lower()

    nim_match = re.search(r'\b\d{10}\b', state['question'])
    
    if nim_match:
        state['idNIMMhs'] = nim_match.group(0)
        cleaned_response = "printktm"
    else:
        cleaned_response = "incompletenim"

    if 'question_type' not in state:
        state['question_type'] = cleaned_response
    else:
        state['question_type'] += f", {cleaned_response}"

    # print(f"question_type: {cleaned_response}\n")
    return {"question_type": cleaned_response}


@time_check
def incompleteNimAgent(state: AgentState):
    info = "\n--- Agent Incomplete NIM ---"
    print(info)
    response = """
        Dari informasi yang ada, belum terdapat nomor NIM (Nomor Induk Mahasiswa) yang diberikan.
        NIM (Nomor Induk Mahasiswa) yang valid dari Undiksha berjumlah 10 digit angka.
        - Format penulisan pesan:
            Cetak KTM [NIM]
        - Contoh penulisan pesan:
            Cetak KTM XXXXXXXXXX
        Kirimkan NIM yang benar pada pesan ini sesuai format dan contoh, agar bisa mencetak Kartu Tanda Mahasiswa (KTM).
    """
    state["responseIncompleteNim"] = response
    state["agentsContext"] = response
    # print(state["responseKTM"])
    return {"agentsContext": state["responseIncompleteNim"]}


@time_check
def printKtmAgent(state: AgentState):
    info = "\n--- Agent Print KTM ---"
    print(info)

    nim_match = re.search(r'\b\d{10}\b', state['question'])
    if nim_match:
        state['idNIMMhs'] = nim_match.group(0)

    id_nim_mhs = state.get("idNIMMhs", "ID NIM tidak berhasil didapatkan.")
    url_nim_mhs = cetak_ktm_mhs(state)
    
    prompt = f"""
        Anda bertugas untuk memberikan gambar Kartu Tanda Mahasiswa (KTM).
        1. NIM milik pengguna: {id_nim_mhs}
        2. Link gambar KTM milik pengguna: {url_nim_mhs}
        Hasilkan respon berupa list kalimat tersebut untuk disampaikan NIM dan Link nya ke pengguna.
    """
    messages = [
        SystemMessage(content=prompt)
    ]
    response = chat_groq(messages)
    state["responseKTM"] = response
    state["agentsContext"] = response
    # print(state["responseKTM"])
    return {"agentsContext": state["responseKTM"]}


@time_check
def outOfContextAgent(state: AgentState):
    info = "\n--- AGENT OUT OF CONTEXT ---"
    print(info)
    response = "Pertanyaan tidak relevan dengan konteks kampus Universitas Pendidikan Ganesha."
    state["responseOutOfContext"] = response
    state["agentsContext"] = response
    # print (state["responseOutOfContext"])
    return {"agentsContext": state["responseOutOfContext"]}


@time_check
def resultWriterAgent(state: AgentState):
    info = "\n--- AGENT RESULT WRITER AGENT ---"
    print(info)
    prompt = f"""
        Berikut pedoman yang harus diikuti untuk menulis ulang informasi:
        - Awali dengan "Salam Harmoni🙏"
        - Anda adalah penulis yang hebat dan pintar.
        - Tugas Anda adalah merangkai informasi secara lengkap dan jelas serta apa adanya sesuai informasi yang diberikan.
        - Jangan mengarang jawaban dari informasi yang diberikan sehingga semua poin penting tersampaikan dan tidak ada yang terlewat.
        Berikut adalah informasinya:
        {state["agentsContext"]}
    """
    messages = [
        SystemMessage(content=prompt)
    ]
    response = chat_openai(messages)
    state["responseFinal"] = response
    print (state["responseFinal"])
    return {"responseFinal": state["responseFinal"]}


@time_check
def build_graph(question):
    workflow = StateGraph(AgentState)
    initial_state = questionIdentifierAgent({"question": question})
    context = initial_state["question_type"]
    workflow.add_node("questionIdentifier", lambda x: initial_state)
    workflow.add_node("resultWriter", resultWriterAgent)
    workflow.add_edge(START, "questionIdentifier")

    if "general" in context:
        workflow.add_node("general", generalAgent)
        workflow.add_node("graderdocs", graderDocsAgent)
        workflow.add_node("answergenerator", answerGeneratorAgent)
        workflow.add_node("graderhallucinations", graderHallucinationsAgent)
        workflow.add_edge("questionIdentifier", "general")
        workflow.add_edge("general", "graderdocs")
        workflow.add_edge("graderdocs", "answergenerator")
        workflow.add_edge("answergenerator", "graderhallucinations")
        workflow.add_conditional_edges(
            "graderhallucinations",
            lambda state: state["generalIsHallucination"], {
                True: "answergenerator",
                False: "resultWriter",
            }
        )

    if "ktm" in context:
        workflow.add_node("ktm", ktmAgent)
        workflow.add_node("incompletenim", incompleteNimAgent)
        workflow.add_node("printktm", printKtmAgent)
        workflow.add_edge("questionIdentifier", "ktm")
        workflow.add_conditional_edges(
            "ktm",
            lambda state: state["question_type"],
            {
                "incompletenim": "incompletenim",
                "printktm": "printktm"
            }
        )
        workflow.add_edge("incompletenim", "resultWriter")
        workflow.add_edge("printktm", "resultWriter")

    if "outofcontext" in context:
        workflow.add_node("outofcontext", outOfContextAgent)
        workflow.add_edge("questionIdentifier", "outofcontext")
        workflow.add_edge("outofcontext", "resultWriter")

    workflow.add_edge("resultWriter", END)

    graph = workflow.compile()
    graph.invoke({'question': question})
    get_graph_image(graph)


build_graph("siapa rektor undiksha dan bagaimana tahapan jadwal SNBP undiksha? Saya juga ingin melihat ktm dengan nim 2115101014 dan gimana caranya agar kaya raya?")
print(f"DEBUG: Total execution time: {total_execution_time:.4f} seconds")