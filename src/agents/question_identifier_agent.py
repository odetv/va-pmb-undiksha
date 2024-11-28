import re
from utils.debug_time import time_check
from utils.expansion import query_expansion, CONTEXT_ABBREVIATIONS
from utils.agent_state import AgentState
from langchain_core.messages import HumanMessage, SystemMessage
from utils.llm import chat_llm


@time_check
def questionIdentifierAgent(state: AgentState):
    info = "\n--- QUESTION IDENTIFIER ---"
    print(info)

    original_question = state['question']
    cleaned_question = re.sub(r'\n+', ' ', original_question)
    expanded_question = query_expansion(cleaned_question, CONTEXT_ABBREVIATIONS)
    state["question"] = expanded_question

    promptTypeQuestion = """
        Anda adalah seoarang pemecah pertanyaan pengguna.
        Tugas Anda sangat penting. Klasifikasikan atau parsing pertanyaan dari pengguna untuk dimasukkan ke variabel sesuai konteks.
        Tergantung pada jawaban Anda, akan mengarahkan ke agent yang tepat.
        Ada 4 konteks diajukan:
        - GENERAL_AGENT - Pertanyaan yang menyebutkan segala informasi umum, penerimaan mahasiswa baru (PMB), perkuliahan kampus baik itu akademik dan mahasiswa, tentang administrasi yang berkaitan dengan dosen pegawai mahasiswa, tentang identitasmu, dan jika ada sapaan maka jawablah.
        - KELULUSAN_AGENT - Pertanyaan terkait pengecekan status kelulusan bagi pendaftaran calon mahasiswa baru yang telah mendaftar di Undiksha, biasanya pertanyaan pengguna berisi nomor pendaftaran dan tanggal lahir.
        - KTM_AGENT - Hanya jika pertanyaan mengandung kata "ktm" atau "nim". Jika menyebutkan "nip" maka itu general.
        - OUTOFCONTEXT_AGENT - Hanya jika diluar dari konteks.
        Kemungkinan pertanyaannya berisi lebih dari 1 variabel konteks yang berbeda (jika hanya 1, maka 1 saja), buat yang sesuai dengan konteks saja (jika tidak ada jangan dibuat).
        Jawab pertanyaan dan sertakan pertanyaan pengguna yang sesuai dengan kategori dengan contoh seperti ({"GENERAL_AGENT": "pertanyaan relevan terkait general", "KELULUSAN_AGENT": "pertanyaan relevan terkait kelulusan", "KTM_AGENT": "hanya jika pertanyaan mengandung kata "ktm" atau "nim", "OUTOFCONTEXT_AGENT": "pertanyaan diluar konteks"}) begitu seterusnya.
        Buat dengan format data JSON tanpa membuat key baru.
    """
    messagesTypeQuestion = [
        SystemMessage(content=promptTypeQuestion),
        HumanMessage(content=expanded_question),
    ]
    responseTypeQuestion = chat_llm(messagesTypeQuestion).strip().lower()
    
    state["question_type"] = responseTypeQuestion
    print("\nPertanyaan:", expanded_question)

    total_agents = 0
    if "general_agent" in state["question_type"]:
        total_agents += 3
    if "kelulusan_agent" in state["question_type"]:
        total_agents += 2
    if "ktm_agent" in state["question_type"]:
        total_agents += 2
    if "outofcontext_agent" in state["question_type"]:
        total_agents += 1
    state["totalAgents"] = total_agents
    print(f"DEBUG: Total agents bertugas: {state['totalAgents']}")
    pattern = r'"(.*?)":\s*"(.*?)"'
    matches = re.findall(pattern, responseTypeQuestion)
    result_dict = {key: value for key, value in matches}

    state["generalQuestion"] = result_dict.get("general_agent", None)
    state["kelulusanQuestion"] = result_dict.get("kelulusan_agent", None)
    state["ktmQuestion"] = result_dict.get("ktm_agent", None)
    state["outOfContextQuestion"] = result_dict.get("outofcontext_agent", None)
    
    print(f"DEBUG: generalQuestion: {state['generalQuestion']}")
    print(f"DEBUG: kelulusanQuestion: {state['kelulusanQuestion']}")
    print(f"DEBUG: ktmQuestion: {state['ktmQuestion']}")
    print(f"DEBUG: outOfContextQuestion: {state['outOfContextQuestion']}")

    return state