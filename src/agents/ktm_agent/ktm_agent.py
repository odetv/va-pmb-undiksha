import re
from langchain_core.messages import HumanMessage, SystemMessage
from utils.agent_state import AgentState
from utils.llm import chat_llm
from utils.debug_time import time_check


@time_check
def ktmAgent(state: AgentState):
    info = "\n--- KTM ---"
    print(info)

    prompt = """
        Anda adalah agen analis informasi Kartu Tanda Mahasiswa (KTM).
        Tugas Anda adalah mengklasifikasikan jenis pertanyaan pada konteks Undiksha (Universitas Pendidikan Ganesha).
        NIM (Nomor Induk Mahasiswa) yang valid dari Undiksha berjumlah 10 digit angka.
        Sekarang tergantung pada jawaban Anda, akan mengarahkan ke agent yang tepat.
        Ada 2 konteks pertanyaan yang diajukan:
        - TRUE - Jika pengguna menyertakan NIM (Nomor Induk Mahasiswa).
        - FALSE - Jika pengguna tidak menyertakan nomor NIM (Nomor Induk Mahasiswa) dan tidak valid.
        Hasilkan hanya 1 sesuai kata (TRUE, FALSE).
    """

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=state["ktmQuestion"]),
    ]
    response = chat_llm(messages).strip().lower()

    nim_match = re.search(r"\b(?:ktm|kartu tanda mahasiswa)\s*.*?(\b\d{10}\b)(?!\d)", state["ktmQuestion"], re.IGNORECASE)
    if nim_match:
        state["nimKTMMhs"] = nim_match.group(1)
        response = "true"
    else:
        response = "false"
    is_complete = response == "true"

    state["checkKTM"] = is_complete
    state["finishedAgents"].add("ktm_agent")

    return {"checkKTM": state["checkKTM"]}


@time_check
def routeKTMAgent(state: AgentState):
    return state["checkKTM"]