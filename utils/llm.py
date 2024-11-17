import os
from langchain_ollama import OllamaLLM
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv


load_dotenv()
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
openai_api_key = os.getenv("OPENAI_API_KEY")
va_llm = os.getenv("VA_LLM_SERVICE")
va_embedder = os.getenv("VA_EMBEDDER_SERVICE")


def chat_llm(question: str):
    if va_llm == "ollama":
        ollama = OllamaLLM(base_url=ollama_base_url, model="gemma2", verbose=True)
        result = ollama.invoke(question)
        return result
    elif va_llm == "openai":
        openai = ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini", temperature=0, streaming=True)
        result = ""
        try:
            stream_response = openai.stream(question)
            for chunk in stream_response:
                token = chunk.content
                result += token
                print(token, end="", flush=True)
        except Exception as e:
            error = str(e)
            if "401" in error and "Incorrect API key" in error:
                raise ValueError("Incorrect API key provided. Please check your OpenAI API key.")
            else:
                raise e
        return result
    else:
        raise ValueError("LLM yang dipilih pada environment tidak valid! Gunakan 'ollama' atau 'openai'.")


def embedder():
    if va_embedder == "ollama":
        MODEL_EMBEDDING = "bge-m3"
        EMBEDDER = OllamaEmbeddings(base_url=ollama_base_url, model=MODEL_EMBEDDING, show_progress=True)
        return MODEL_EMBEDDING, EMBEDDER
    elif va_embedder == "openai":
        MODEL_EMBEDDING = "text-embedding-3-large"
        EMBEDDER = OpenAIEmbeddings(api_key=openai_api_key, model=MODEL_EMBEDDING)
        return MODEL_EMBEDDING, EMBEDDER
    else:
        raise ValueError("EMBEDDER yang dipilih pada environment tidak valid! Gunakan 'ollama' atau 'openai'.")