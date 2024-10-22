from typing import List, Set, TypedDict
from langchain.memory import ConversationBufferMemory


class AgentState(TypedDict):
    context: str
    question: str
    question_type: str
    generalContext: str
    generalGraderDocs: str
    generalIsHallucination: str
    responseGeneral: str
    checkKTM: str
    responseKTM: str
    responseIncompleteNim: str
    checkKelulusan: str
    responseKelulusan: str
    responseIncompleteInfoKelulusan: str
    responseOutOfContext: str
    responseFinal: str
    idNIMMhs: str
    urlKTMMhs: str
    noPendaftaran: str
    pinPendaftaran: str
    finishedAgents: Set[str]
    agentsContext: str
    memory: ConversationBufferMemory