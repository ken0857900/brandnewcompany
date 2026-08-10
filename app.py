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

# Day5：News Agent，搜新聞 → 摘要 → 輸出 JSON {summary, bullish, confidence}
# ponytail: 單一新聞源（Tavily），多來源整合/罐頭文字過濾（參考 HOYA BIT fetch_news.py）留到需要時再補
class NewsState(TypedDict):
    question: str
    news: str
    report: str

def news_agent(state: NewsState) -> dict:
    reply = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[{
            "role": "user",
            "content": (
                "根據以下新聞搜尋結果，輸出一個 JSON，欄位固定是："
                "summary（一句話中文摘要）、bullish（true 或 false，這則新聞對後市偏多還是偏空）、"
                "confidence（0 到 1 的信心分數）。\n"
                f"新聞搜尋結果：\n{state['news']}"
            ),
        }],
    )
    report = reply.choices[0].message.content
    parsed = json.loads(report)  # 自我檢查：格式壞了這裡就直接炸，不會讓壞資料流到下游
    assert {"summary", "bullish", "confidence"} <= parsed.keys(), f"News Agent 輸出缺欄位: {parsed}"
    return {"report": report}

news_builder = StateGraph(NewsState)
news_builder.add_node("search", search)
news_builder.add_node("news_agent", news_agent)
news_builder.add_edge(START, "search")
news_builder.add_edge("search", "news_agent")
news_builder.add_edge("news_agent", END)
news_app = news_builder.compile()

print("輸入 exit 結束；輸入「news 幣種」觸發 News Agent（例：news BTC）")
while True:
    q = input("You: ")
    if q.lower() == "exit":
        break
    if q.lower().startswith("news "):
        coin = q[5:].strip()
        result = news_app.invoke({"question": f"{coin} 最新新聞"})
        print(result["report"])
        continue
    result = app.invoke({"question": q})
    print(result["answer"])


