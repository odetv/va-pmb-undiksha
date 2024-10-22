import time
from utils.agent_state import AgentState


def time_check(func):
    def wrapper(state: AgentState):
        start_time = time.time()
        result = func(state)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"DEBUG: {func.__name__} took {execution_time:.4f} seconds\n\n")
        return result
    return wrapper