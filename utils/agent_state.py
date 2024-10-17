from typing import TypedDict
from langchain.memory import ConversationBufferMemory

class AgentState(TypedDict):
    context: str
    question: str
    question_type: str
    generalContext: str
    generalGraderDocs: str
    generalIsHallucination: str
    agentsContext: str
    responseGeneral: str
    responseKTM: str
    responseIncompleteNim: str
    responseOutOfContext: str
    responseFinal: str
    idNIMMhs: str
    urlNIMMhs: str
    memory: ConversationBufferMemory
