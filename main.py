import re
from langgraph.graph import END, START, StateGraph
from typing import TypedDict
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, SystemMessage
from tools.llm import chat_openai, chat_ollama
from tools.apiUndiksha import apiKtmMhs


class AgentState(TypedDict):
    context : str
    question : str
    question_type : str
    nimMhs: str
    memory: ConversationBufferMemory


def questionIdentifierAgent(state: AgentState):
    info = "--- QUESTION IDENTIFIER ---"
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
    response = chat_ollama(messages)
    cleaned_response = response.strip().lower()
    print("Pertanyaan:", state["question"])
    print(f"question_type: {cleaned_response}\n")
    return {"question_type": cleaned_response}


def generalAgent(state: AgentState):
    info = "--- GENERAL ---"
    print(info+"\n")
    return "Informasi umum mengenai Undiksha."


def ktmAgent(state: AgentState):
    info = "--- KTM ---"
    print(info)
    prompt = """
        Anda adalah seoarang analis informasi Kartu Tanda Mahasiswa (KTM).
        Tugas Anda adalah mengklasifikasikan jenis pertanyaan pada konteks Undiksha (Universitas Pendidikan Ganesha).
        ID atau NIM (Nomor Induk Mahasiswa) yang valid dari Undiksha berjumlah 10 digit angka.
        Sekarang tergantung pada jawaban Anda, akan mengarahkan ke agent yang tepat.
        Ada 2 konteks pertanyaan yang diajukan:
        - INCOMPLETEID - Jika pengguna tidak menyertakan nomor ID atau NIM (Nomor Induk Mahasiswa) dan tidak valid.
        - PRINTKTM - Jika pengguna menyertakan nomor ID atau NIM.
        Hasilkan hanya 1 sesuai kata (INCOMPLETEID, PRINTKTM).
    """
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=state["question"]),
    ]
    response = chat_ollama(messages)
    cleaned_response = response.strip().lower()

    nim_match = re.search(r'\b\d{10}\b', state['question'])
    
    if nim_match:
        state['nimMhs'] = nim_match.group(0)
        cleaned_response = "printktm"
    else:
        cleaned_response = "incompleteid"

    if 'question_type' not in state:
        state['question_type'] = cleaned_response
    else:
        state['question_type'] += f", {cleaned_response}"

    print(f"question_type: {cleaned_response}\n")
    return {"question_type": cleaned_response}


def inCompleteIdAgent(state: AgentState):
    info = "--- INCOMPLETE ID ---"
    print(info+"\n")
    prompt = f"""
        Anda adalah validator yang hebat dan pintar.
        Tugas Anda adalah memvalidasi ID atau NIM (Nomor Induk Mahasiswa) pada konteks Undiksha (Universitas Pendidikan Ganesha).
        Dari informasi yang ada, belum terdapat nomor ID atau NIM (Nomor Induk Mahasiswa) yang diberikan.
        ID atau NIM (Nomor Induk Mahasiswa) yang valid dari Undiksha berjumlah 10 digit angka.
        - Format penulisan pesan:
            Cetak KTM [NIM]
        - Contoh penulisan pesan:
            Cetak KTM 2115XXXXXX
        Hasilkan respon untuk meminta pengguna kirimkan NIM yang benar pada pesan ini sesuai format dan contoh, agar bisa mencetak Kartu Tanda Mahasiswa (KTM).
    """
    messages = [
        SystemMessage(content=prompt)
    ]
    response = chat_ollama(messages)
    print(response)
    return response


def printKtmAgent(state: AgentState, urlKtmMhs):
    info = "--- PRINT KTM ---"
    print(info+"\n")
    nimMhs = state.get('nimMhs', 'NIM tidak ditemukan')
    apiKtmMhs
    prompt = f"""
        Anda adalah tukang memberikan Kartu Tanda Mahasiswa (KTM).
        - NIM milik pengguna: {nimMhs}
        - Link gambar KTM milik pengguna: {urlKtmMhs}
        Hasilkan respon berupa kalimat yang mengatakan ini KTM milikmu dan ini link preview gambar Kartu Tanda Mahasiswa (KTM).
    """
    messages = [
        SystemMessage(content=prompt)
    ]
    response = chat_ollama(messages)
    print(response)
    return response


def outOfContextAgent(state: AgentState):
    info = "--- OUT OF CONTEXT ---"
    print(info+"\n")
    return "Pertanyaan tidak relevan dengan konteks kampus."


def resultWriterAgent(state: AgentState, agent_results):
    info = "--- RESULT WRITER AGENT ---"
    print(info+"\n")
    prompt = f"""
        Awali dengan "Salam Harmoni🙏"
        Anda adalah penulis yang hebat dan pintar.
        Tugas Anda adalah merangkai jawaban yang jelas dan komprehensif berdasarkan informasi yang diberikan.
        Jangan mengarang jawaban dari informasi yang diberikan.
        Berikut adalah informasi dari berbagai sumber:
        {agent_results}
        Susun ulang informasi ini dan memodifikasi agar lebih koheren dan mudah dipahami.
        Gunakan penomoran, URL, link atau yang lainnya jika diperlukan.
        Pastikan semua poin penting tersampaikan dan tidak ada yang terlewat, jangan mengatakan proses penyusunan ulang ini.

    """
    messages = [
        SystemMessage(content=prompt)
    ]
    response = chat_ollama(messages)
    print(response)
    return response


def executeAgents(state: AgentState, agents):
    agent_results = []
    while agents:
        agent = agents.pop(0)
        if agent == "general":
            agent_results.append(generalAgent(state))
        elif agent == "ktm":
            # agent_results.append(ktmAgent(state))
            ktm_result = ktmAgent(state)
            additional_agents = routeToSpecificAgent(state)
            for additional_agent in additional_agents:
                if additional_agent not in agents:
                    agents.insert(0, additional_agent)
        elif agent == "incompleteid":
            agent_results.append(inCompleteIdAgent(state))
        elif agent == "printktm":
            # agent_results.append(printKtmAgent(state))
            urlKtmMhs = apiKtmMhs()
            agent_results.append(printKtmAgent(state, urlKtmMhs))
        elif agent == "outOfContext":
            agent_results.append(outOfContextAgent(state))
    print(f"Konteks: {agent_results}\n")
    return agent_results


def routeToSpecificAgent(state: AgentState):
    question_types = [q_type.strip().lower() for q_type in state["question_type"].split(",")]
    agents = []
    if "general" in question_types:
        agents.append("general")
    if "ktm" in question_types:
        agents.append("ktm")
    if "incompleteid" in question_types:
        agents.append("incompleteid")
    if "printktm" in question_types:
        agents.append("printktm")
    if "outofcontext" in question_types:
        agents.append("outOfContext")
    return agents


# Definisikan Langgraph
workflow = StateGraph(AgentState)

# Definisikan Node
workflow.add_node("question_identifier", questionIdentifierAgent)
workflow.add_node("general", generalAgent)
workflow.add_node("ktm", ktmAgent)
workflow.add_node("inCompleteId", inCompleteIdAgent)
workflow.add_node("printKtm", printKtmAgent)
workflow.add_node("outOfContext", outOfContextAgent)
workflow.add_node("resultWriter", resultWriterAgent)

# Definisikan Edge
workflow.add_edge(START, "question_identifier")
workflow.add_conditional_edges(
    "question_identifier",
    routeToSpecificAgent
)

graph = workflow.compile()


# Contoh pertanyaan
question = "Cetak KTM 2115XXXXXX"
state = {"question": question}

# Jalankan question identifier untuk mendapatkan agen yang perlu dieksekusi
question_identifier_result = questionIdentifierAgent(state)

# Identifikasi agen-agen yang relevan
agents_to_execute = routeToSpecificAgent(question_identifier_result)

# Eksekusi semua agen yang relevan dan kumpulkan hasilnya
agent_results = executeAgents(state, agents_to_execute)

# Jalankan resultWriterAgent untuk menghasilkan jawaban final
resultWriterAgent(state, agent_results)