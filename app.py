import json
from dotenv import load_dotenv
from openai import OpenAI
from tavily import TavilyClient

load_dotenv()
client = OpenAI()
tavily = TavilyClient()

tools = [{
    "type": "function",
    "function": {
        "name": "search_news",
        "description": "Search the web for current news and information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"}
            },
            "required": ["query"],
        },
    },
}]

messages = [{"role": "user", "content": "BTC 今天有什麼新聞？"}]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools,
)
message = response.choices[0].message

if message.tool_calls:
    tool_call = message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    print(f"[Claude 決定呼叫 search_news，查詢：{args['query']}]")

    result = tavily.search(args["query"])

    messages.append(message)
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(result, ensure_ascii=False),
    })

    followup = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
    )
    print(followup.choices[0].message.content)
else:
    print(message.content)
