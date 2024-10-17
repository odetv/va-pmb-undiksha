import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.agent_state import AgentState


def cetak_ktm_mhs(state: AgentState):
    id_nim_mhs = state.get("idNIMMhs")
    url_nim_mhs = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTOwRConBYl2t6L8QMOAQqa5FDmPB_bg7EnGA&s"
    state["urlNIMMhs"] = url_nim_mhs
    return url_nim_mhs