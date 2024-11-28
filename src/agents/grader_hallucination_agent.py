from langchain_core.messages import SystemMessage
from utils.agent_state import AgentState
from utils.llm import chat_llm
from utils.debug_time import time_check


@time_check
def graderHallucinationAgent(state: AgentState):
    info = "\n--- GRADER HALLUCINATION ---"
    print(info)

    if "responseFinal" not in state:
        state["responseFinal"] = ""
    if "hallucinationCount" not in state:
        state["hallucinationCount"] = 0

    prompt = f"""
        Anda adalah seorang penilai dari OPINI dengan FAKTA.
        Berikan nilai "false" hanya jika OPINI ada kaitannya dengan FAKTA atau berikan nilai "true" hanya jika OPINI tidak ada kaitannya dengan FAKTA.
        Harap cermat dalam menilai, karena ini akan sangat bergantung pada jawaban Anda.
        - OPINI: {state["responseFinal"]}
        - FAKTA: {state["answerAgents"]}
    """

    messages = [
        SystemMessage(content=prompt)
    ]
    response = chat_llm(messages).strip().lower()
    is_hallucination = response == "true"
    state["isHallucination"] = is_hallucination

    if is_hallucination:
        state["hallucinationCount"] += 1
    else:
        state["hallucinationCount"] = 0

    state["isHallucination"] = is_hallucination
    print(f"Apakah hasil halusinasi? {is_hallucination}")
    print(f"Jumlah pengecekan halusinasi berturut-turut: {state['hallucinationCount']}")

    return {"isHallucination": state["isHallucination"], "hallucinationCount": state["hallucinationCount"]}