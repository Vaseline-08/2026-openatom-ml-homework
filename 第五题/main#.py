import os
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise ValueError("没有读取到 DEEPSEEK_API_KEY，请检查 .env 文件")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)

# 核心提示词：强制模型按固定格式输出
SYSTEM_PROMPT = """你是一个校园信息助手。
请根据用户输入的校园通知或问题，严格按以下格式输出，不要添加任何额外说明：

一句话总结：
关键信息：
待办事项：

要求：
1. 只依据用户提供的文本，不要编造原文中没有的信息。
2. 如果原文没有时间、地点或截止日期，在对应位置写“未提及”。
3. 待办事项如果原文中没有明确任务，写“无明确待办”。
"""

def ask_model(user_input: str) -> str:
    """发送用户输入给模型，返回格式化回答"""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input}
            ],
            temperature=0.3,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"发生错误：{e}"

def main():
    print("=== 校园信息助手 ===")
    print("输入校园通知或问题，模型会按固定格式输出。")
    print("输入 q 或 quit 退出程序。\n")

    while True:
        user_input = input("请输入内容：\n").strip()

        if user_input.lower() in ("q", "quit"):
            print("已退出，再见！")
            break

        if not user_input:
            print("输入不能为空，请重新输入。\n")
            continue

        print("\n--- 模型输出 ---")
        answer = ask_model(user_input)
        print(answer)
        print("--- 输出结束 ---\n")
        # 假设 answer = response.choices[0].message.content

        print("--- 模型输出 ---")
        print(answer)

        # 改进点1：程序侧的二次校验
        warning_keywords = ["未提及", "无明确待办", "暂无", "没有提供"]
        has_warning = False

        for kw in warning_keywords:
            if kw in answer:
                has_warning = True
                break

        if has_warning:
            print("\n⚠️ 提示：原始文本可能未提供完整的时间、地点或明确任务，请以正式通知为准！")

if __name__ == "__main__":
    main()