from operator import add
from typing_extensions import TypedDict, Annotated, Sequence, Set


class AnswerState(TypedDict):
    question = None
    answer = None


class AgentState(TypedDict):
    context: str
    question: str
    question_type: str
    generalQuestion: str
    kelulusanQuestion: str
    ktmQuestion: str
    outOfContextQuestion: str
    totalAgents: int
    finishedAgents: Set[str]
    generalContext: str
    generalGraderDocs: str
    checkKelulusan: str
    noPendaftaran: str
    tglLahirPendaftar: str
    checkKTM: str
    nimKTMMhs: str
    urlKTMMhs: str
    answerAgents : Annotated[Sequence[AnswerState], add]
    retrievedContext : str
    responseFinal: str
    isHallucination: str
    hallucinationCount: int