from utils.agent_state import AgentState
from utils.debug_time import time_check


@time_check
def incompleteInfoKelulusanAgent(state: AgentState):
    info = "\n--- Incomplete Info Kelulusan SMBJM ---"
    print(info)

    response = """
        Untuk mengecek kelulusan dapat melalui link https://penerimaan.undiksha.ac.id/smbjm/login atau langsung dari pesan ini saya dapat membantu anda, namun diperlukan mengirimkan Nomor Pendaftaran dan Tanggal Lahir Pendaftar SMBJM.
        - Format penulisan pesan:
            Cek Kelulusan Nomor Pendaftaran [NO_PENDAFTARAN_10_DIGIT] Tanggal Lahir [YYYY-MM-DD]
        - Contoh penulisan pesan:
            Cek Kelulusan Nomor Pendaftaran 1234567890 Tanggal Lahir 2001-01-31
    """

    agentOpinion = {
        "question": state["kelulusanQuestion"],
        "answer": response
    }
    state["finishedAgents"].add("incompleteInfoKelulusan_agent")

    return {"answerAgents": [agentOpinion]}