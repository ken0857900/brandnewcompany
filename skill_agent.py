"""
Day 12 — Skill Topology
一個 agent，依產業動態載入不同 rubric／口吻檔（純文字檔）
"""
from pathlib import Path
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware.types import dynamic_prompt, ModelRequest

load_dotenv()

RUBRIC_DIR = Path(__file__).parent / "rubrics"
INDUSTRIES = ["服飾", "配件", "美妝", "其他"]

# ---------- 1. 口吻檔：純文字，不進 git 也能改，agent 不用重寫 ----------
DEFAULT_RUBRICS = {
    "服飾": """產業：服飾
語氣：活潑、有畫面感，像朋友推薦戰袍
句長：短句為主，一句不超過 20 字
用詞：可以用「穿上就自帶氣場」「顯瘦」「單品混搭」
禁忌：不要用「精緻」「質感生活」這種美妝式詞彙""",
    "配件": """產業：配件
語氣：俐落、聚焦細節與搭配情境
句長：中等句長，可以列 2-3 個使用情境
用詞：可以用「畫龍點睛」「日常百搭」「一秒升級造型」
禁忌：不要過度誇飾材質（沒認證別說「精鋼」「真皮」）""",
    "美妝": """產業：美妝
語氣：溫柔、強調膚感與使用體驗
句長：可以稍長，描述質地與上妝後的感覺
用詞：可以用「服貼」「輕透」「一擦提亮」
禁忌：不要用醫療級療效字眼（治療、消除，避免誇大廣告用語）""",
    "其他": """產業：其他（不屬於服飾／配件／美妝的商品）
語氣：中性、直接介紹商品本身的功能與特色，不強加特定產業的情緒詞
句長：中等句長，講清楚是什麼、能幹嘛、適合誰用
用詞：避免借用其他產業的專屬詞彙（不要硬套「顯瘦」「膚感」這類詞）
禁忌：不要為了套用某個產業的語氣，硬把商品講成它不是的東西""",
}


def _ensure_rubric_files() -> None:
    """rubrics/ 資料夾不存在或缺檔就補上預設版，之後你直接改 txt 內容即可"""
    RUBRIC_DIR.mkdir(exist_ok=True)
    for name, content in DEFAULT_RUBRICS.items():
        f = RUBRIC_DIR / f"{name}.txt"
        if not f.exists():
            f.write_text(content, encoding="utf-8")


# ---------- 2. 分類：跟 Day 11 同一個判斷，但只回一個字串，不再選 tool ----------
_classifier = init_chat_model("gpt-4o-mini")


def classify_industry(product_desc: str) -> str:
    resp = _classifier.invoke(
        f"判斷這個商品屬於哪個產業，只回答「服飾」「配件」「美妝」「其他」四個詞其中一個，"
        f"不確定或都不屬於前三類就回答「其他」，不要多餘文字：\n{product_desc}"
    )
    industry = resp.content.strip()
    if industry not in INDUSTRIES:
        raise ValueError(f"分類結果不合法：模型回了「{industry}」，不在 {INDUSTRIES} 裡")
    return industry


# ---------- 3. 單一 agent + dynamic prompt middleware ----------
def _build_rubric_prompt(industry: str) -> str:
    rubric = (RUBRIC_DIR / f"{industry}.txt").read_text(encoding="utf-8")
    return f"你是商品文案寫手，依照以下規則撰寫文案：\n\n{rubric}"


@dynamic_prompt
def load_rubric(request: ModelRequest) -> str:
    industry = request.runtime.context.get("industry", "服飾")
    return _build_rubric_prompt(industry)


copywriter = create_agent(
    model="gpt-4o-mini",
    tools=[],
    middleware=[load_rubric],
)


def write_copy(product_desc: str) -> str:
    industry = classify_industry(product_desc)
    print(f"（分類結果：{industry}）")
    result = copywriter.invoke(
        {"messages": [{"role": "user", "content": f"幫這個商品寫文案：{product_desc}"}]},
        context={"industry": industry},
    )
    return result["messages"][-1].content


# ---------- 4. 手動測三種：直接輸入，輸入空白結束 ----------
if __name__ == "__main__":
    _ensure_rubric_files()
    while True:
        desc = input("\n輸入商品描述（直接 Enter 結束）：").strip()
        if not desc:
            break
        print(write_copy(desc))
