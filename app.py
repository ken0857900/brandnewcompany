import json
from dotenv import load_dotenv
from openai import OpenAI
from tavily import TavilyClient
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from datetime import datetime

load_dotenv()
client = OpenAI()
tavily = TavilyClient()

class State(TypedDict):
    question: str
    news: str
    answer: str

def search(state: State) -> dict:
    result = tavily.search(state["question"])
    return {"news": json.dumps(result, ensure_ascii=False)}

def answer(state: State) -> dict:
    today = datetime.now().strftime("%Y-%m-%d")
    reply = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"今天是 {today}。根據這些搜尋結果回答「{state['question']}」：\n{state['news']}"}],
    )
    return {"answer": reply.choices[0].message.content}

graph = StateGraph(State)
graph.add_node("search", search)
graph.add_node("answer", answer)
graph.add_edge(START, "search")
graph.add_edge("search", "answer")
graph.add_edge("answer", END)
app = graph.compile()

print("輸入 exit 結束")
while True:
    q = input("You: ")
    if q.lower() == "exit":
        break
    result = app.invoke({"question": q})
    print(result["answer"])


