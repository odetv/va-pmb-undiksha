from utils.agent_state import AgentState
from utils.debug_time import time_check


@time_check
def outOfContextAgent(state: AgentState):
    info = "\n--- OUT OF CONTEXT ---"
    print(info)

    response = """
        Pertanyaan tidak relevan dengan konteks kampus Universitas Pendidikan Ganesha.
    """

    agentOpinion = {
        "question": state["outOfContextQuestion"],
        "answer": response
    }
    state["finishedAgents"].add("outofcontext_agent")

    return {"answerAgents": [agentOpinion]}