"""
src.agents.agent — Construction de l'agent LangGraph
"""

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict

from src.llms import get_llm
from src.tools import get_tools
from src.prompts import get_system_prompt


# ─── État de l'agent ──────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ─── Nœud principal — appel au LLM ───────────────────────────────────────────
def call_model(state: AgentState) -> dict:
    llm = get_llm()
    tools = get_tools()
    system_prompt = get_system_prompt()

    llm_with_tools = llm.bind_tools(tools)
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# ─── Construction du graphe ───────────────────────────────────────────────────
def build_agent():
    """Construit et compile le graphe LangGraph de l'agent."""
    graph = StateGraph(AgentState)

    graph.add_node("agent", call_model)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)

    return graph.compile()


# ─── Exécution directe (test) ─────────────────────────────────────────────────
if __name__ == "__main__":
    agent = build_agent()
    result = agent.invoke({"messages": [HumanMessage(content="Bonjour, qui es-tu ?")]})
    print(result["messages"][-1].content)
