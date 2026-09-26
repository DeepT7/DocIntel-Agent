import os
import operator
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from google import genai
from langgraph.graph import StateGraph, START, END

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY")
if not API_KEY:
    raise SystemExit("API_KEY not found. Please set GEMINI_API_KEY or API_KEY in your .env file.")

client = genai.Client(api_key=API_KEY)


class State(TypedDict):
    messages: Annotated[list[str], operator.add]
    llm_calls: int


def llm_call(state: State) -> dict:
    user_message = state["messages"][-1] if state["messages"] else ""
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=user_message,
    )

    return {
        "messages": [response.text],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


workflow = StateGraph(State)
workflow.add_node("llm_call", llm_call)
workflow.add_edge(START, "llm_call")
workflow.add_edge("llm_call", END)

app = workflow.compile()


if __name__ == "__main__":
    user_question = input("Ask a question: ").strip()
    if not user_question:
        print("Please enter a question.")
    else:
        result = app.invoke({"messages": [user_question], "llm_calls": 0})
        print("\nAnswer:\n", result["messages"][-1])
