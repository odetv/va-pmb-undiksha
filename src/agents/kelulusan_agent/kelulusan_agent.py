import re
from langchain_core.messages import HumanMessage, SystemMessage
from utils.agent_state import AgentState
from utils.llm import chat_llm
from utils.debug_time import time_check


@time_check
def kelulusanAgent(state: AgentState):
    info = "\n--- CEK KELULUSAN SMBJM ---"
    print(info)

    prompt = """
        Anda adalah seoarang analis informasi kelulusan SMBJM.
        Tugas Anda adalah mengklasifikasikan jenis pertanyaan pada konteks Undiksha (Universitas Pendidikan Ganesha).
        Sekarang tergantung pada jawaban Anda, akan mengarahkan ke agent yang tepat.
        Ada 2 konteks pertanyaan yang diajukan:
        - TRUE - Jika pengguna menyertakan Nomor Pendaftaran (Format 10 digit angka) dan Tanggal Lahir (Format YYYY-MM-DD).
        - FALSE - Jika pengguna tidak menyertakan Nomor Pendaftaran (Format 10 digit angka) dan Tanggal Lahir (Format YYYY-MM-DD).
        Hasilkan hanya 1 sesuai kata (TRUE, FALSE).
    """
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=state["kelulusanQuestion"]),
    ]
    response = chat_llm(messages).strip().lower()

    noPendaftaran_match = re.search(r"\b(?:nmr|no|nomor|nmr.|no.|nomor.|nmr. |no. |nomor. )\s*pendaftaran.*?(\b\d{10}\b)(?!\d)", state["kelulusanQuestion"], re.IGNORECASE)
    tglLahirPendaftar_match = re.search(r"(?:ttl|tanggal lahir|tgl lahir|lahir|tanggal-lahir|tgl-lahir|lhr|tahun|tahun lahir|thn lahir|thn|th lahir)[^\d]*(\d{4}-\d{2}-\d{2})", state["kelulusanQuestion"], re.IGNORECASE)

    if noPendaftaran_match and tglLahirPendaftar_match:
        state["noPendaftaran"] = noPendaftaran_match.group(1)
        state["tglLahirPendaftar"] = tglLahirPendaftar_match.group(1)
        response = "true"
    else:
        response = "false"
    is_complete = response == "true"

    state["checkKelulusan"] = is_complete
    state["finishedAgents"].add("kelulusan_agent") 
    print(f"Info Kelulusan Lengkap? {is_complete}")

    return {"checkKelulusan": state["checkKelulusan"]}


@time_check
def routeKelulusanAgent(state: AgentState):
    return state["checkKelulusan"]