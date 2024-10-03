from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os


load_dotenv()
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
openai_api_key = os.getenv("OPENAI_API_KEY")


def chat_ollama(question: str, model = "gemma2"):
    ollama = Ollama(base_url=ollama_base_url, model=model, verbose=True)
    result = ollama.invoke(question)
    return result


def chat_openai(question: str):
    openai = ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini")
    result = openai.invoke(question).content if hasattr(openai.invoke(question), "content") else openai.invoke(question)
    return result