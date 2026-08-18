from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

load_dotenv()

agent = create_agent(
    model=ChatOpenAI(model="gpt-4o-mini"),
    tools=[],
    system_prompt="你是一間 AI 投研公司的助理。",
)

result = agent.invoke({"messages": [{"role": "user", "content": "你好，你是誰？"}]})
print(result["messages"][-1].content)
