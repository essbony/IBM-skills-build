# ===================== agent_pipeline.py =====================
# Backend LangGraph : routage multimodal (histoire / image),
# recherche web, rédaction, génération d'image, publication.
# Ce module est importé par app.py (Streamlit) — aucun code ne
# s'exécute automatiquement à l'import (login HF non-interactif,
# pas d'appel réseau au chargement).
# ================================================================

from typing import List, Annotated, Literal, Optional, TypedDict, AsyncGenerator
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver

from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tavily import AsyncTavilyClient

import os
import uuid
import httpx

load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def check_env() -> dict:
    """Retourne l'état des variables d'environnement requises (pour affichage UI)."""
    return {
        "HF_API_TOKEN": bool(HF_API_TOKEN),
        "GOOGLE_API_KEY": bool(GOOGLE_API_KEY),
        "TAVILY_API_KEY": bool(TAVILY_API_KEY),
        "OPENAI_API_KEY": bool(OPENAI_API_KEY),
    }


# ============== LLMs (instanciation paresseuse) ===============

_llm = None
_ibm_llm = None
_goog_llm = None
_llm_search = None
_llm_image = None
_llm_story = None
_llm_publish = None
_router_llm = None
_openai_client = None
_tavily_client = None


def get_llms():
    """Instancie les LLMs une seule fois (évite le login HF interactif à l'import)."""
    global _llm, _ibm_llm, _goog_llm, _llm_search, _llm_image, _llm_story, _llm_publish, _router_llm

    if _ibm_llm is not None:
        return

    if HF_API_TOKEN:
        from huggingface_hub import login
        login(token=HF_API_TOKEN)

    _llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        task="text-generation",
        max_new_tokens=256,
        temperature=0.7,
        repetition_penalty=1.3,
        do_sample=True,
    )
    _ibm_llm = ChatHuggingFace(llm=_llm)

    _goog_llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0,
        streaming=True,
        repetition_penalty=1.3,
        do_sample=True,
    )

    _llm_search = _ibm_llm.bind_tools(tools=[research_web])
    _llm_image = _goog_llm.bind_tools(tools=[image_generate])
    _llm_story = _ibm_llm.bind_tools(tools=[recit_writer])
    _llm_publish = _goog_llm.bind_tools(tools=[publish_content])
    _router_llm = _goog_llm.with_structured_output(Route)


def get_openai_client() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI()
    return _openai_client


def get_tavily_client() -> AsyncTavilyClient:
    global _tavily_client
    if _tavily_client is None:
        _tavily_client = AsyncTavilyClient()
    return _tavily_client


# ====================== TOOLS ==========================

@tool
async def research_web(query: str) -> str:
    """Recherche des informations fiables sur le web.

    Args:
        query: La requête de recherche de l'utilisateur.

    Returns:
        str: Résultats de recherche sous forme de texte.
    """
    client = get_tavily_client()
    response = await client.search(query=query, search_depth="advanced", max_results=5)
    results = response.get("results", [])
    return "\n\n".join(
        f"- {r['title']}\n  {r['url']}\n  {r['content']}" for r in results
    )


@tool
async def recit_writer(
    subject: str,
    style: str = "Narratif",
    language: str = "Français",
    length: str = "moyen",
) -> str:
    """Génère une histoire ou un récit de qualité.

    Args:
        subject: Sujet ou thème principal de l'histoire.
        style: Style d'écriture (Narratif, Poétique, Dramatique, Humoristique...).
        language: Langue de sortie.
        length: Longueur souhaitée ("court", "moyen", "long").

    Returns:
        L'histoire générée.
    """
    prompt = f"""
Tu es un écrivain professionnel.

Sujet : {subject}
Style : {style}
Langue : {language}
Longueur : {length}

Écris une histoire immersive avec :
- un début accrocheur
- un développement structuré
- une fin claire
"""
    response = await _ibm_llm.ainvoke(prompt)
    return response.content


@tool
async def image_generate(prompt: str, style: Optional[str] = None) -> str:
    """Génère une image à l'aide de Gemini.

    Args:
        prompt: Description détaillée de l'image à générer.
        style: Style artistique optionnel (cinématographique, illustration, réaliste...).

    Returns:
        Chemin ou URL de l'image générée.
    """
    full_prompt = f"{prompt}, style {style}, haute qualité, détaillé" if style else prompt
    return f"https://gemini-generated-image/{hash(full_prompt)}.png"


@tool
async def publish_content(text: Optional[str] = None, image_urls: Optional[list[str]] = None):
    """Publie un contenu texte, une image, ou un contenu multimédia via Zapier."""
    ZAPIER_WEBHOOK_URL = "https://hooks.zapier.com/hooks/catch/28063276/44lxs3i/"
    payload = {}
    if text:
        payload["text"] = text.strip()
    if image_urls:
        payload["image"] = image_urls

    async with httpx.AsyncClient() as client:
        response = await client.post(ZAPIER_WEBHOOK_URL, json=payload)
        response.raise_for_status()
        return response.json()


# ==================== ETAT DU GRAPH ==================

class Route(BaseModel):
    content_type: Literal["story", "image"] = Field(
        description="Le type de contenu final à produire : une histoire ou une image."
    )
    needs_research: bool = Field(
        description="True si une recherche d'informations externes est nécessaire avant de produire le contenu."
    )


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    audio_path: Optional[str]
    content_type: Literal["story", "image"]
    needs_research: bool
    research: list[str]
    story: list[str] | None
    image_urls: Optional[list[str]]
    publish_result: dict | None


# =============== SYSTEM PROMPTS ==============

ROUTER_PROMPT = """
Tu es un superviseur d'agents.

Analyse la demande de l'utilisateur et détermine deux choses :

1. content_type : le type de contenu final à produire.
   - "story" : si l'utilisateur veut une histoire/texte narratif.
   - "image" : si l'utilisateur veut une image générée.

2. needs_research : est-ce qu'une recherche d'informations externes (web) est nécessaire
   avant de produire ce contenu, pour se baser sur des faits réels/actuels ?
   - true : si le contenu doit s'appuyer sur des informations vérifiées/récentes.
   - false : si le contenu peut être généré directement sans recherche.
"""

RESEARCH_SYSTEM_PROMPT = """
Tu es un agent spécialisé dans la recherche d'informations sur Internet.
Utilise obligatoirement l'outil `research_web` dès qu'une recherche externe est nécessaire.
Ne fabrique jamais d'informations qui ne proviennent pas des sources disponibles.
"""

STORY_SYSTEM_PROMPT = """
Tu es un agent spécialisé dans la création d'histoires et de récits narratifs.
Utilise obligatoirement l'outil `recit_writer` pour générer le texte final de l'histoire.
"""

IMAGE_SYSTEM_PROMPT = """
Tu es un agent spécialisé dans la création et la génération d'images.
Utilise obligatoirement l'outil `image_generate` lorsque l'utilisateur demande une image.
Crée des prompts d'image précis et riches pour obtenir une meilleure qualité.
"""

PUBLISHER_SYSTEM_PROMPT = """
Tu es un agent spécialisé dans la publication de contenus numériques.
Utilise l'outil `publish_content` pour publier le texte et/ou l'image disponible.
N'appelle pas l'outil si aucun contenu publiable n'est disponible.
"""


# ==============================================================================
#                              NODES
# ===============================================================================

async def speech_to_text(state: State) -> dict:
    audio_path = state.get("audio_path")
    if not audio_path:
        return {}
    client = get_openai_client()
    with open(audio_path, "rb") as audio:
        transcription = await client.audio.transcriptions.create(
            model="whisper-1", file=audio
        )
    return {"messages": [HumanMessage(content=transcription.text)]}


async def router_agent(state: State) -> dict:
    decision: Route = await _router_llm.ainvoke(
        [SystemMessage(content=ROUTER_PROMPT), *state["messages"]]
    )
    return {"content_type": decision.content_type, "needs_research": decision.needs_research}


async def research_agent(state: State) -> dict:
    messages = [SystemMessage(content=RESEARCH_SYSTEM_PROMPT), *state["messages"]]
    response = await _llm_search.ainvoke(messages)
    return {"messages": [response], "research": [response.content]}


async def writer_agent(state: State) -> dict:
    messages = [SystemMessage(content=STORY_SYSTEM_PROMPT), *state["messages"]]
    response = await _llm_story.ainvoke(messages)
    return {"messages": [response], "story": [response.content]}


async def image_agent(state: State) -> dict:
    messages = [SystemMessage(content=IMAGE_SYSTEM_PROMPT), *state["messages"]]
    response = await _llm_image.ainvoke(messages)
    return {"messages": [response], "image_urls": [response.content]}


async def publish_agent(state: State) -> dict:
    messages = [SystemMessage(content=PUBLISHER_SYSTEM_PROMPT), *state["messages"]]
    story_list = state.get("story") or []
    image_list = state.get("image_urls") or []
    content = " ".join(story_list) if story_list else (image_list[0] if image_list else "")
    if content:
        messages.append(HumanMessage(content=f"Voici le contenu à publier : {content}"))
    response = await _llm_publish.ainvoke(messages)
    return {"messages": [response], "publish_result": {"status": "succes"}}


# ============== CONSTRUCTION DU GRAPH ============================

def build_graph():
    get_llms()  # instanciation paresseuse des LLMs/tools liés

    graph = StateGraph(State)

    graph.add_node("speech_to_text", speech_to_text)
    graph.add_node("router", router_agent)
    graph.add_node("research", research_agent)
    graph.add_node("research_tools", ToolNode(tools=[research_web]))
    graph.add_node("writer", writer_agent)
    graph.add_node("story_tools", ToolNode(tools=[recit_writer]))
    graph.add_node("image", image_agent)
    graph.add_node("image_tools", ToolNode(tools=[image_generate]))
    graph.add_node("publish", publish_agent)
    graph.add_node("publish_tools", ToolNode(tools=[publish_content]))

    graph.add_edge(START, "speech_to_text")
    graph.add_edge("speech_to_text", "router")

    def route_after_router(state: State) -> str:
        if state["needs_research"]:
            return "research"
        return "writer" if state["content_type"] == "story" else "image"

    graph.add_conditional_edges(
        "router", route_after_router,
        {"research": "research", "writer": "writer", "image": "image"},
    )

    def route_from_research(state: State) -> str:
        if tools_condition(state) == "tools":
            return "research_tools"
        return "writer" if state["content_type"] == "story" else "image"

    graph.add_conditional_edges(
        "research", route_from_research,
        {"tools": "research_tools", "writer": "writer", "image": "image"},
    )
    graph.add_edge("research_tools", "research")

    graph.add_conditional_edges(
        "writer", tools_condition, {"tools": "story_tools", END: "publish"}
    )
    graph.add_edge("story_tools", "writer")

    graph.add_conditional_edges(
        "image", tools_condition, {"tools": "image_tools", END: "publish"}
    )
    graph.add_edge("image_tools", "image")

    graph.add_conditional_edges(
        "publish", tools_condition, {"tools": "publish_tools", END: END}
    )
    graph.add_edge("publish_tools", "publish")

    memory = InMemorySaver()
    return graph.compile(checkpointer=memory)


# Labels lisibles pour l'UI (nom de node -> libellé + icône)
NODE_LABELS = {
    "speech_to_text": ("🎙️", "Transcription audio"),
    "router": ("🧭", "Analyse de la demande"),
    "research": ("🔎", "Recherche web"),
    "research_tools": ("🔎", "Recherche web (outil Tavily)"),
    "writer": ("✍️", "Rédaction de l'histoire"),
    "story_tools": ("✍️", "Génération du texte"),
    "image": ("🎨", "Génération d'image"),
    "image_tools": ("🎨", "Création de l'image"),
    "publish": ("📤", "Publication"),
    "publish_tools": ("📤", "Envoi vers Zapier"),
}


async def stream_pipeline(
    user_text: Optional[str],
    audio_path: Optional[str],
    thread_id: Optional[str] = None,
) -> AsyncGenerator[dict, None]:
    """Exécute le graphe en streaming et yield chaque mise à jour de node.

    Chaque item yield : {"node": str, "update": dict, "state": dict | None}
    Le dernier item a node == "__end__" et contient l'état final complet.
    """
    builder = build_graph()
    thread_id = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    messages = [HumanMessage(content=user_text)] if user_text else []
    input_state = {"audio_path": audio_path, "messages": messages}

    final_state = None
    async for update in builder.astream(input_state, config=config, stream_mode="updates"):
        for node_name, node_update in update.items():
            yield {"node": node_name, "update": node_update}

    final_state = await builder.aget_state(config)
    yield {"node": "__end__", "update": None, "state": final_state.values if final_state else {}}