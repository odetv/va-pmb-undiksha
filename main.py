from langgraph.graph import END, START, StateGraph
from utils.agent_state import AgentState
from utils.debug_time import time_check
from utils.create_graph_image import get_graph_image
from src.agents.question_identifier_agent import questionIdentifierAgent
from src.agents.result_writer_agent import resultWriterAgent
from src.agents.grader_hallucination_agent import graderHallucinationAgent
from src.agents.general_agent.general_agent import generalAgent
from src.agents.general_agent.grader_docs_general_agent import graderDocsGeneralAgent
from src.agents.general_agent.answer_general_agent import answerGeneralAgent
from src.agents.kelulusan_agent.kelulusan_agent import kelulusanAgent, routeKelulusanAgent
from src.agents.kelulusan_agent.incomplete_info_kelulusan_agent import incompleteInfoKelulusanAgent
from src.agents.kelulusan_agent.info_kelulusan_agent import infoKelulusanAgent
from src.agents.ktm_agent.ktm_agent import ktmAgent, routeKTMAgent
from src.agents.ktm_agent.incomplete_info_ktm_agent import incompleteInfoKTMAgent
from src.agents.ktm_agent.info_ktm_agent import infoKTMAgent
from src.agents.outofcontext_agent.outofcontext_agent import outOfContextAgent


@time_check
def rag_adaptive(question):
    workflow = StateGraph(AgentState)
    initial_state = questionIdentifierAgent({"question": question, "finishedAgents": set()})
    context = initial_state["question_type"]
    workflow.add_node("questionIdentifier_agent", lambda x: initial_state)
    workflow.add_edge(START, "questionIdentifier_agent")

    if "general_agent" in context:
        workflow.add_node("general_agent", generalAgent)
        workflow.add_node("graderDocsGeneral_agent", graderDocsGeneralAgent)
        workflow.add_node("answerGeneral_agent", answerGeneralAgent)
        workflow.add_edge("questionIdentifier_agent", "general_agent")
        workflow.add_edge("general_agent", "graderDocsGeneral_agent")
        workflow.add_edge("graderDocsGeneral_agent", "answerGeneral_agent")
        workflow.add_edge("answerGeneral_agent", "resultWriter_agent")

    if "kelulusan_agent" in context:
        workflow.add_node("kelulusan_agent", kelulusanAgent)
        workflow.add_node("incompleteInfoKelulusan_agent", incompleteInfoKelulusanAgent)
        workflow.add_node("infoKelulusan_agent", infoKelulusanAgent)
        workflow.add_edge("questionIdentifier_agent", "kelulusan_agent")
        workflow.add_conditional_edges(
            "kelulusan_agent",
            routeKelulusanAgent,
            {
                True: "infoKelulusan_agent",
                False: "incompleteInfoKelulusan_agent"
            }
        )
        workflow.add_edge("incompleteInfoKelulusan_agent", "resultWriter_agent")
        workflow.add_edge("infoKelulusan_agent", "resultWriter_agent")

    if "ktm_agent" in context:
        workflow.add_node("ktm_agent", ktmAgent)
        workflow.add_node("incompleteInfoKTM_agent", incompleteInfoKTMAgent)
        workflow.add_node("infoKTM_agent", infoKTMAgent)
        workflow.add_edge("questionIdentifier_agent", "ktm_agent")
        workflow.add_conditional_edges(
            "ktm_agent",
            routeKTMAgent,
            {
                True: "infoKTM_agent",
                False: "incompleteInfoKTM_agent"
            }
        )
        workflow.add_edge("incompleteInfoKTM_agent", "resultWriter_agent")
        workflow.add_edge("infoKTM_agent", "resultWriter_agent")

    if "outofcontext_agent" in context:
        workflow.add_node("outofcontext_agent", outOfContextAgent)
        workflow.add_edge("questionIdentifier_agent", "outofcontext_agent")
        workflow.add_edge("outofcontext_agent", "resultWriter_agent")

    workflow.add_node("resultWriter_agent", resultWriterAgent)
    workflow.add_node("graderHallucination_agent", graderHallucinationAgent)
    workflow.add_edge("resultWriter_agent", "graderHallucination_agent")
    workflow.add_conditional_edges(
        "graderHallucination_agent",
        lambda state: state["isHallucination"] and state["hallucinationCount"] < 2 if state["isHallucination"] else False,
        {
            True: "resultWriter_agent",
            False: END,
        }
    )

    graph = workflow.compile()
    result = graph.invoke({"question": question})
    answers = result.get("responseFinal", [])
    contexts = result.get("retrievedContext", [])
    # contexts = result.get("answerAgents", "")
    get_graph_image(graph)
    return contexts, answers



# DEBUG QUERY EXAMPLES
# rag_adaptive("Apa jenis-jenis beasiswa bagi mahasiswa undiksha?")