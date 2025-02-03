from utils.debug_time import time_check
from utils.agent_state import AgentState
from langchain_community.vectorstores import FAISS
from utils.llm import embedder
from src.config.config import VECTORDB_DIR


@time_check
def generalAgent(state: AgentState):
    info = "\n--- GENERAL ---"
    print(info)

    VECTOR_PATH = VECTORDB_DIR
    _,EMBEDDER = embedder()
    question = state["generalQuestion"]
    try:
        vectordb = FAISS.load_local(VECTOR_PATH, EMBEDDER, allow_dangerous_deserialization=True)
        retriever = vectordb.similarity_search(question, k=5)
        context = "\n\n\n\n".join([f"Konteks ke-{i+1}\n{doc.page_content}" for i, doc in enumerate(retriever)])
        print (context)
        state["retrievedContext"] = context

    except RuntimeError as e:
        if "could not open" in str(e):
            raise RuntimeError("Vector database FAISS index file not found. Please ensure the index file exists at the specified path.")
        else:
            raise

    state["generalContext"] = context
    state["finishedAgents"].add("general_agent")

    return state