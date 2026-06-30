"""
UI/chainlit.py — Interface conversationnelle avec Chainlit
Pour lancer : chainlit run UI/chainlit.py
"""

import chainlit as cl
from src.agents import build_agent
from langchain_core.messages import HumanMessage


agent = build_agent()


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("thread_id", cl.context.session.id)
    await cl.Message(content="👋 Bonjour ! Je suis votre agent IA. Comment puis-je vous aider ?").send()


@cl.on_message
async def on_message(message: cl.Message):
    thread_id = cl.user_session.get("thread_id")
    config = {"configurable": {"thread_id": thread_id}}

    response = agent.invoke(
        {"messages": [HumanMessage(content=message.content)]},
        config=config,
    )

    answer = response["messages"][-1].content
    await cl.Message(content=answer).send()
