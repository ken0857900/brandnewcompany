"""
Day 14 — Handoff Topology
main agent 判斷客戶回應是不是「嫌太正式」→ 是的話才把控制權轉給改寫 agent
"""
from dotenv import load_dotenv
from langchain.agents import create_agent

load_dotenv()

formal_agent = create_agent(
    model="gpt-4o-mini",
    tools=[],
    system_prompt="你是正式商品文案師，用專業、正式的語氣寫文案，強調材質與規格。",
)

judge_agent = create_agent(
    model="gpt-4o-mini",
    tools=[],
    system_prompt="你負責判斷客戶對商品文案的回應，是不是在說文案太正式、太生硬、想要更口語一點。只回答「是」或「否」，不要多餘文字。",
)

casual_agent = create_agent(
    model="gpt-4o-mini",
    tools=[],
    system_prompt="你是口語商品文案師，用輕鬆、朋友聊天的語氣改寫文案，短句、可以用表情符號。",
)


def is_too_formal(feedback: str) -> bool:
    """main agent 判斷，不是關鍵字比對——客戶不管怎麼講，只要意思是嫌太正式就算"""
    result = judge_agent.invoke({"messages": [{"role": "user", "content": feedback}]})
    return result["messages"][-1].content.strip() == "是"


if __name__ == "__main__":
    desc = input("輸入商品描述：").strip()
    result = formal_agent.invoke(
        {"messages": [{"role": "user", "content": f"幫這個商品寫文案：{desc}"}]}
    )
    print("\n正式版文案：", result["messages"][-1].content)

    while True:
        feedback = input("\n輸入客戶回應（直接 Enter 結束）：").strip()
        if not feedback:
            break
        if is_too_formal(feedback):
            print("（main agent 判斷：太正式，觸發 handoff）")
            handoff_messages = result["messages"] + [{"role": "user", "content": feedback}]
            rewritten = casual_agent.invoke({"messages": handoff_messages})
            print("改寫後文案：", rewritten["messages"][-1].content)
        else:
            print("（main agent 判斷：不是嫌太正式，不觸發 handoff）")
            break
