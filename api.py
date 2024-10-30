from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from main import build_graph


app = FastAPI()
class QuestionRequest(BaseModel):
    question: str
class QuestionResponse(BaseModel):
    question: str
    answer: str


@app.post("/chat", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    question = request.question
    try:
        answer = build_graph(question)
        if not answer:
            raise HTTPException(status_code=500, detail="No answer generated.")
        return QuestionResponse(question=question, answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# DEBUG
# uvicorn api:app --reload
