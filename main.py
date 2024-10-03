from langgraph.graph import END, START, StateGraph
from typing import TypedDict, Optional
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, SystemMessage
from llm import chat_ollama, chat_openai


class AgentState(TypedDict):
    context : str
    question_type : str
    question : str
    memory: ConversationBufferMemory


def questionIdentifierAgent(state: AgentState):
    print("--- QUESTION IDENTIFIER AGENT ---")
    print(state["question"])
    prompt = """
        Anda adalah analis pertanyaan pengguna. Tugas anda adalah mengklasifikasikan pertanyaan yang masuk.
        Tergantung pada jawaban Anda, pertanyaan akan diarahkan ke tim yang tepat, jadi tugas Anda sangat penting.
        Ingat, Anda perlu memvalidasi pertanyaan harus pada konteks di Universitas Pendidikan Ganesha (Undiksha) saja.
        Perhatikan pertanyaan yang diberikan, harus cek pertanyaan agar spesifik lengkap sesuai konteks.
        Ada 3 kemungkinan konteks pertanyaan yang diajukan:
        - GENERAL - Pertanyaan terkait informasi seputar kampus, baik itu akademik dan mahasiswa di Undiksha, serta Penerimaan Mahasiswa Baru di Undiksha.
        - KTM - Pertanyaan terkait Kartu Tanda Mahasiswa (KTM).
        - OUT OF CONTEXT - Hanya jika tidak tahu jawabannya berdasarkan konteks yang diberikan.
        Hasilkan hanya satu kata (GENERAL, KTM, OUT OF CONTEXT).
    """
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=state["question"]),
    ]
    response = chat_openai(messages)
    print(f"question_type: {response}")
    return {"question_type": response}


def routeToSpecificAgent(state: AgentState): 
    return state["question_type"]


def generalAgent(state: AgentState):
    print("--- GENERAL AGENT ---")
    pass


def ktmAgent(state: AgentState):
    print("--- KTM AGENT ---")
    pass


def outOfContextAgent(state: AgentState):
    print("--- OUT OF CONTEXT AGENT ---")
    pass


# Definisikan Langgraph
workflow = StateGraph(AgentState)

# Definisikan Node
workflow.add_node("question_identifier", questionIdentifierAgent)
workflow.add_node("general", generalAgent)
workflow.add_node("ktm", ktmAgent)
workflow.add_node("outOfContext", outOfContextAgent)

# Definisikan Edge
workflow.add_edge(START, "question_identifier")
workflow.add_conditional_edges(
    "question_identifier",
    routeToSpecificAgent, {
        "GENERAL": "general",
        "KTM": "ktm",
        "OUT OF CONTEXT": "outOfContext",
        # "GENERAL \n": "general",
        # "KTM \n": "ktm",
        # "OUT OF CONTEXT \n": "outOfContext",
    }
)

# Compile Graph
graph = workflow.compile()


question = "gimana caranya daftar di undiksha?"
graph.invoke({"question": question})