from langchain_core.messages import HumanMessage, SystemMessage
from utils.agent_state import AgentState
from utils.llm import chat_llm
from utils.debug_time import time_check


@time_check
def resultWriterAgent(state: AgentState):
    if len(state["finishedAgents"]) < state["totalAgents"]:
        print("\nMenunggu agent lain menyelesaikan tugas...")
        return None
    
    elif len(state["finishedAgents"]) == state["totalAgents"]:
        info = "\n--- RESULT WRITER ---"
        print(info)

        prompt = f"""
            Berikut pedoman yang harus diikuti untuk menulis ulang informasi:
            - Awali dengan "Salam Harmoni🙏"
            - Berikan informasi secara lengkap dan jelas apa adanya sesuai informasi yang diberikan.
            - Urutan informasi sesuai dengan urutan pertanyaan.
            - Jangan menyebut ulang pertanyaan secara eksplisit.
            - Jangan tawarkan informasi lainnya selain konteks yang didapat saja.
            - Hasilkan response dalam format Markdown.
            Berikut adalah informasinya:
            {state["answerAgents"]}
        """

        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=state["question"])
        ]
        response = chat_llm(messages)
        
        state["responseFinal"] = response
        
        return {"responseFinal": state["responseFinal"]}