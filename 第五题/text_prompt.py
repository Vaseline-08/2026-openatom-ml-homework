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
SYSTEM_PROMPT = """总结以下通知
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
    print("=== 提示词对比：简单提示词版 ===")
    print("本版本的系统提示词只有“总结以下通知”，输出格式不固定，用于与 main.py 对照。")
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

if __name__ == "__main__":
    main()
