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

messages = []  # ← 記憶放這裡，跨輪都不清空

print("輸入 exit 結束")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    messages.append({"role": "user", "content": user_input})

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
        reply = followup.choices[0].message
        print(reply.content)
        messages.append(reply)
    else:
        print(message.content)
        messages.append(message)
