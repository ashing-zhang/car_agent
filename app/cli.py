# AutoAgent CLI - 交互式命令行演示(规格第22节)
# 运行指南:
#   python -m app.cli
#   输入 "quit" 或 "exit" 退出
#   示例: "现在车速多少?" / "有点冷" / "把温度调到24度"

import logging

from langchain_core.messages import AIMessage, HumanMessage

from app.agent.graph import build_default_agent

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def run_repl() -> None:
    """运行交互式问答循环。"""
    agent = build_default_agent()
    print("=" * 60)
    print("AutoAgent CLI - 车载智能助手(输入 quit 退出)")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break

        result = agent.invoke(
            {
                "messages": [HumanMessage(content=user_input)],
                "user_id": "demo-user",
                "session_id": "cli-session",
            }
        )
        messages = result["messages"]

        tool_lines: list[str] = []
        final_response = ""
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    args_repr = ", ".join(f"{k}={v}" for k, v in tc["args"].items())
                    tool_lines.append(f"Tool > {tc['name']}({args_repr})")
            if isinstance(msg, AIMessage) and msg.content and msg is messages[-1]:
                final_response = msg.content

        for line in tool_lines:
            print(line)
        if final_response:
            print(f"Agent > {final_response}")
        elif not tool_lines:
            print("Agent > (无回复)")


def main() -> None:
    """CLI 入口。"""
    run_repl()


if __name__ == "__main__":
    main()
