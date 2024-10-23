from typing import TypedDict, Annotated, Sequence, Set
from operator import add
from langchain.memory import ConversationBufferMemory


class AnswerState(TypedDict):
    agent = None
    answer = None

class AgentState(TypedDict):
    context: str
    question: str
    question_type: str
    generalContext: str
    generalGraderDocs: str
    generalIsHallucination: str
    responseGeneral: str
    checkKelulusan: str
    noPendaftaran: str
    pinPendaftaran: str
    responseIncompleteInfoKelulusan: str
    responseKelulusan: str
    checkKTM: str
    idNIMMhs: str
    urlKTMMhs: str
    responseIncompleteNim: str
    responseKTM: str
    responseOutOfContext: str
    responseFinal: str
    finishedAgents: Set[str]
    answerAgents : Annotated[Sequence[AnswerState], add]
    memory: ConversationBufferMemory