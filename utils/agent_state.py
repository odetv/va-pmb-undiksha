from operator import add
from typing_extensions import TypedDict, Annotated, Sequence, Set
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
    tglLahirPendaftar: str
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
    generalQuestion: str
    kelulusanQuestion: str
    ktmQuestion: str
    outOfContextQuestion: str
    memory: ConversationBufferMemory