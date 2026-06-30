"""
main.py — Point d'entrée principal du projet
"""

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from src.agents import build_agent

load_dotenv()


def main():
    print("🤖 Agent IA — project-bob démarré\n")
    agent = build_agent()
    thread_id = "main-session"

    while True:
        user_input = input("Vous : ").strip()
        if not user_input or user_input.lower() in {"exit", "quit", "q"}:
            print("Au revoir !")
            break

        config = {"configurable": {"thread_id": thread_id}}
        response = agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )
        print(f"Agent : {response['messages'][-1].content}\n")


if __name__ == "__main__":
    main()
