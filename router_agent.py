from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini")

clothing_agent = create_agent(model=model, tools=[], system_prompt="你是服飾文案師，強調版型、材質觸感、穿搭情境。")
accessory_agent = create_agent(model=model, tools=[], system_prompt="你是配件文案師，強調細節做工、百搭性、送禮情境。")
beauty_agent = create_agent(model=model, tools=[], system_prompt="你是美妝文案師，強調成分、膚感、使用步驟。")

@tool
def write_clothing_copy(product_description: str) -> str:
    """商品屬於服飾類時使用，寫服飾文案。輸入商品描述文字。"""
    result = clothing_agent.invoke({"messages": [{"role": "user", "content": product_description}]})
    return result["messages"][-1].content

@tool
def write_accessory_copy(product_description: str) -> str:
    """商品屬於配件類時使用，寫配件文案。輸入商品描述文字。"""
    result = accessory_agent.invoke({"messages": [{"role": "user", "content": product_description}]})
    return result["messages"][-1].content

@tool
def write_beauty_copy(product_description: str) -> str:
    """商品屬於美妝類時使用，寫美妝文案。輸入商品描述文字。"""
    result = beauty_agent.invoke({"messages": [{"role": "user", "content": product_description}]})
    return result["messages"][-1].content

router_agent = create_agent(
    model=model,
    tools=[write_clothing_copy, write_accessory_copy, write_beauty_copy],
    system_prompt="你是產業分類兼派工員。讀商品描述，判斷屬於服飾、配件、還是美妝，只呼叫對應的那一個工具，不要同時呼叫多個。呼叫完工具後，把工具回傳的內容原封不動貼出來，不要自己再摘要",
)

products = [
    "亮金色月亮花朵戒指，鑲綠色寶石",
    "碎花雪紡洋裝，夏季輕薄",
    "霧面唇釉，顯色持久",
]

for desc in products:
    result = router_agent.invoke({"messages": [{"role": "user", "content": desc}]})
    print(result["messages"][-1].content)
    print("---")
