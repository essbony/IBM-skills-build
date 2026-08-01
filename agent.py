# ===================== IMPORTS =====================

from __future__ import annotations
from typing_extensions import TypedDict
from typing import Annotated, Any, AsyncGenerator, Literal, Optional, Sequence

import os
import uuid
import operator
import logging
import urllib.parse

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.messages import (AIMessage,BaseMessage,HumanMessage,SystemMessage,ToolMessage)
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace



# ============================= CONFIGS ============================

load_dotenv()
logger = logging.getLogger(__name__)
HF_API_TOKEN = os.getenv("HF_API_TOKEN") 
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def check_env() -> dict[str, bool]:
    """État des variables d'environnement (pour l'UI)."""
    return {
        "HF_API_TOKEN": bool(HF_API_TOKEN),
        "GOOGLE_API_KEY": bool(GOOGLE_API_KEY),
        "TAVILY_API_KEY": bool(TAVILY_API_KEY),
        "OPENAI_API_KEY": bool(OPENAI_API_KEY),
    }

# ================== INITIALISATION LLMs ==============

_ibm_llm = None
_goog_llm = None
_llm_search = None
_llm_image = None
_llm_story = None
_llm_publish = None
_router_llm = None
_openai_client = None
_tavily_client = None


def get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import AsyncOpenAI
        _openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def get_tavily_client():
    global _tavily_client
    if _tavily_client is None:
        from tavily import AsyncTavilyClient
        _tavily_client = AsyncTavilyClient(api_key=TAVILY_API_KEY)
    return _tavily_client


def get_llms() -> None:
    """Instancie une seule fois les LLMs et les runnables liés aux tools."""
    global _ibm_llm, _goog_llm
    global _llm_search, _llm_image, _llm_story, _llm_publish, _router_llm

    if _ibm_llm is not None:
        return


    if HF_API_TOKEN:
        try:
            from huggingface_hub import login
            login(token=HF_API_TOKEN, add_to_git_credential=False)
        except Exception as e:
            logger.warning("HF login skipped: %s", e)

   

    hf_endpoint = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        task="text-generation",
        max_new_tokens=1024,
        temperature=0.7,
        repetition_penalty=1.15,
        do_sample=True,
        huggingfacehub_api_token=HF_API_TOKEN,
    )
    _ibm_llm = ChatHuggingFace(llm=hf_endpoint)



    _goog_llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0.2,
        google_api_key=GOOGLE_API_KEY,
    )


    _llm_search = _ibm_llm.bind_tools([research_web])
    _llm_story = _ibm_llm.bind_tools([recit_writer])
    _llm_image = _goog_llm.bind_tools([image_generate])
    _llm_publish = _goog_llm.bind_tools([publish_content])
    _router_llm = _goog_llm.with_structured_output(Route)

# ===================== LES TOOLS ====================

@tool
async def research_web(query: str) -> str:
    """Recherche des informations fiables sur le web (Tavily).

    Args:
        query: Requête de recherche.
    """
    if not TAVILY_API_KEY:
        return "Erreur : TAVILY_API_KEY manquante."
    try:
        client = get_tavily_client()
        response = await client.search(
            query=query, search_depth="advanced", max_results=5
        )
        results = response.get("results", [])
        if not results:
            return "Aucun résultat trouvé."
        return "\n\n".join(
            f"- {r.get('title', '')}\n  {r.get('url', '')}\n  {r.get('content', '')}"
            for r in results
        )
    except Exception as e:
        logger.exception("research_web failed")
        return f"Erreur de recherche : {e}"


@tool
async def recit_writer(
    subject: str,
    style: str = "Narratif",
    language: str = "Français",
    length: str = "moyen",
) -> str:
    """Génère une histoire / récit de qualité.

    Args:
        subject: Thème principal.
        style: Style (Narratif, Poétique, Dramatique, Humoristique…).
        language: Langue de sortie.
        length: court | moyen | long.
    """
    get_llms()
    prompt = f"""Tu es un écrivain professionnel.

Sujet : {subject}
Style : {style}
Langue : {language}
Longueur : {length}

Écris une histoire immersive avec :
- un début accrocheur
- un développement structuré
- une fin claire

Ne mets pas de préambule, commence directement l'histoire."""
    try:
        response = await _ibm_llm.ainvoke(
            [HumanMessage(content=prompt)]
        )
        return response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        logger.exception("recit_writer failed")
        return f"Erreur de génération d'histoire : {e}"


@tool
async def image_generate(prompt: str, style: Optional[str] = None) -> str:
    """Génère une image à partir d'une description et retourne une URL valide.

    Args:
        prompt: Description détaillée de l'image en anglais.
        style: Style optionnel.
    """
    full_prompt = f"{prompt}, style {style}" if style else prompt
    ## Nettoyage et encodage strict du prompt pour l'URL
    clean_prompt = full_prompt.replace("\n", " ").strip()
    encoded_prompt = urllib.parse.quote(clean_prompt)
    
    ## URL directe Pollinations
    image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true"
    logger.info("URL Image générée : %s", image_url)
    return image_url


@tool
async def publish_content(
    text: Optional[str] = None,
    image_urls: Optional[list[str]] = None,
) -> str:
    """Publie un contenu texte et/ou image via un webhook Zapier.

    Args:
        text: Texte à publier.
        image_urls: Liste d'URLs d'images.
    """
    import httpx

    ZAPIER_WEBHOOK_URL = os.getenv("ZAPIER_WEBHOOK_URL")
    payload: dict[str, Any] = {}
    if text and text.strip():
        payload["text"] = text.strip()
    if image_urls:
        payload["image"] = image_urls

    if not payload:
        return "Rien à publier (texte et images vides)."

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(ZAPIER_WEBHOOK_URL, json=payload)
            response.raise_for_status()
            return f"Publié avec succès : {response.json()}"
    except Exception as e:
        logger.exception("publish_content failed")
        return f"Erreur de publication : {e}"


# ======================== ETAT DU GRAPH ET STRAUCTURE DE SORTIE =====================

class Route(BaseModel):
    content_type: Literal["story", "image"] = Field(
        description="Type de contenu final : 'story' ou 'image'."
    )
    needs_research: bool = Field(
        description="True si une recherche web est nécessaire avant de produire le contenu."
    )


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    audio_path: Optional[str]
    content_type: Optional[Literal["story", "image"]]
    needs_research: Optional[bool]
    research: Annotated[list[str], operator.add]
    story: Annotated[list[str], operator.add]
    image_urls: Annotated[list[str], operator.add]
    publish_result: Optional[dict]

# =====================PROMPTS SYSTEMES =======================


ROUTER_PROMPT = """Tu es un superviseur d'agents.

Analyse la demande de l'utilisateur et détermine :

1. content_type :
   - "story" → si la demande concerne l'écriture d'un texte, d'un conte ou d'un récit.
   - "image" → si l'utilisateur demande de générer, créer, dessiner ou illustrer une image/photo/visuel.

2. needs_research :
   - true  → UNIQUEMENT pour les histoires ("story") qui nécessitent des faits récents ou des informations réelles vérifiées.
   - false → TOUJOURS false pour le type "image" (la génération d'image est 100% créative) ou pour les textes purement fictifs.
"""

RESEARCH_SYSTEM_PROMPT = """Tu es un agent de recherche web.
Utilise OBLIGATOIREMENT l'outil `research_web` pour répondre.
Ne fabrique jamais d'informations."""

STORY_SYSTEM_PROMPT = """Tu es un agent de création d'histoires.
Utilise OBLIGATOIREMENT l'outil `recit_writer` pour produire le texte final.
Si des résultats de recherche sont fournis dans le contexte, utilise-les."""

IMAGE_SYSTEM_PROMPT = """Tu es un générateur d'images. Ta SEULE et UNIQUE tâche est d'appeler l'outil `image_generate` "
        "avec une description courte et concise uniquement en ANGLAIS (max 30 mots).\n"
        "RÈGLES STRICTES :\n"
        "- Ne génère AUCUN texte en français.\n"
        "- Ne décris pas l'image dans ton message textuel.\n"
        "- Appelle directement l'outil `image_generate(prompt=...)`."""

PUBLISHER_SYSTEM_PROMPT = """Tu es un agent de publication.
Utilise l'outil `publish_content` avec le texte et/ou les URLs d'images disponibles.
N'appelle pas l'outil s'il n'y a rien à publier."""


# ============================ AIDES ===========================

def _last_ai_content(messages: Sequence[BaseMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.content and not m.tool_calls:
            return str(m.content)
    return ""


def _extract_tool_results(messages: Sequence[BaseMessage]) -> list[str]:
    return [
        str(m.content)
        for m in messages
        if isinstance(m, ToolMessage) and m.content
    ]

# =========================== LES NOEUDS ===========================


async def speech_to_text(state: State) -> dict:
    audio_path = state.get("audio_path")
    if not audio_path:
        return {}
    if not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY manquante → skip transcription")
        return {}
    try:
        client = get_openai_client()
        with open(audio_path, "rb") as audio:
            transcription = await client.audio.transcriptions.create(
                model="whisper-1", file=audio
            )
        return {"messages": [HumanMessage(content=transcription.text)]}
    except Exception as e:
        logger.exception("speech_to_text failed")
        return {"messages": [HumanMessage(content=f"[Erreur transcription] {e}")]}


async def router_agent(state: State) -> dict:
    get_llms()
    decision: Route = await _router_llm.ainvoke(
        [SystemMessage(content=ROUTER_PROMPT), *state["messages"]]
    )
    return {
        "content_type": decision.content_type,
        "needs_research": decision.needs_research,
    }


async def research_agent(state: State) -> dict:
    get_llms()
    messages = [SystemMessage(content=RESEARCH_SYSTEM_PROMPT), *state["messages"]]
    response = await _llm_search.ainvoke(messages)
    return {"messages": [response]}


async def writer_agent(state: State) -> dict:
    get_llms()
    research_ctx = "\n".join(state.get("research") or [])
    extra = []
    if research_ctx:
        extra.append(
            HumanMessage(
                content=f"Voici des informations de recherche à utiliser :\n{research_ctx}"
            )
        )
    messages = [
        SystemMessage(content=STORY_SYSTEM_PROMPT),
        *state["messages"],
        *extra,
    ]
    response = await _llm_story.ainvoke(messages)
    update: dict = {"messages": [response]}
## Si le LLM a déjà répondu en texte (sans tool call), on le stocke
    if isinstance(response, AIMessage) and response.content and not response.tool_calls:
        update["story"] = [str(response.content)]
    return update


async def image_agent(state: State) -> dict:
    get_llms()
    
    if state.get("image_urls"):
        return {}

    messages = [
        SystemMessage(content=IMAGE_SYSTEM_PROMPT),
        *state["messages"],
    ]
    response = await _llm_image.ainvoke(messages)
    return {"messages": [response]}


async def publish_agent(state: State) -> dict:
    get_llms()
    story_parts = state.get("story") or []
    image_list = state.get("image_urls") or []

## Récupération de secours des URLs dans l'historique des messages 
    if not image_list:
        for m in state["messages"]:
            if isinstance(m, ToolMessage) and ("http://" in str(m.content) or "https://" in str(m.content)):
                image_list.append(str(m.content).strip())

    if not story_parts:
        for m in state["messages"]:
            if isinstance(m, ToolMessage) and m.name == "recit_writer":
                story_parts.append(str(m.content))

## Préparation du contenu à transmettre à l'outil Zapier
    content_hint = ""
    if story_parts:
        content_hint += f"Texte : {' '.join(story_parts)}\n"
    if image_list:
        content_hint += f"URL de l'image à publier : {image_list[0]}\n"

    system_prompt = (
        "Tu es l'agent de publication Zapier. Tu dois OBLIGATOIREMENT appeler l'outil `publish_content`. "
        "Transmets le texte dans l'argument `text` et l'URL de l'image dans l'argument `image_urls`."
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Publie ces éléments immédiatement :\n{content_hint}"),
    ]

    response = await _llm_publish.ainvoke(messages)
    return {
        "messages": [response],
        "image_urls": image_list,  # S'assure que le State conserve l'URL pour Streamlit
        "publish_result": {"status": "sent"},
    }


# ====================== CONSTRUCTION DU GRAPH =================

def build_graph():
    get_llms()

    graph = StateGraph(State)

    graph.add_node("speech_to_text", speech_to_text)
    graph.add_node("router", router_agent)
    graph.add_node("research", research_agent)
    graph.add_node("research_tools", ToolNode([research_web]))
    graph.add_node("writer", writer_agent)
    graph.add_node("story_tools", ToolNode([recit_writer]))
    graph.add_node("image", image_agent)
    graph.add_node("image_tools", ToolNode([image_generate]))
    graph.add_node("publish", publish_agent)
    graph.add_node("publish_tools", ToolNode([publish_content]))

## START → speech → router
    def route_from_start(state: State) -> str:
        if state.get("audio_path"):
            return "speech_to_text"
        return "router"

    graph.add_conditional_edges(
        START,
        route_from_start,
        {
            "speech_to_text": "speech_to_text",
            "router": "router",
        }
    )
    graph.add_edge("speech_to_text", "router")
    graph.add_edge(START, "speech_to_text")
    graph.add_edge("speech_to_text", "router")

    def route_after_router(state: State) -> str:
        ## Si c'est une image, on ignore la recherche web et on va directement à l'image
        if state.get("content_type") == "image":
            return "image"
        if state.get("needs_research"):
            return "research"
        return "writer"

    graph.add_conditional_edges(
        "router",
        route_after_router,
        {"research": "research", "writer": "writer", "image": "image"},
    )

## Research loop
    def route_from_research(state: State) -> str:
        if tools_condition(state) == "tools":
            return "research_tools"
        return "writer" if state.get("content_type") == "story" else "image"

    graph.add_conditional_edges(
        "research",
        route_from_research,
        {
            "research_tools": "research_tools",
            "writer": "writer",
            "image": "image",
        },
    )
    graph.add_edge("research_tools", "research")

## Après research_tools on reboucle sur research ; quand research
## n'a plus de tool_calls on va vers writer/image.
## On capture aussi les ToolMessage dans le state.research

    async def capture_research(state: State) -> dict:
        results = _extract_tool_results(state["messages"])
        return {"research": results} if results else {}

## On insère un petit nœud de capture juste avant d'aller writer/image
## (alternative simple : le faire dans route_from_research via un side-effect
##  n'est pas possible, donc on laisse les ToolMessage dans messages
##  et writer/image les lisent via state["research"] alimenté ailleurs).

    # Writer loop
    graph.add_conditional_edges(
        "writer",
        tools_condition,
        {"tools": "story_tools", "__end__": "publish"},
    )
    graph.add_edge("story_tools", "writer")

    # Image loop
    graph.add_conditional_edges(
        "image",
        tools_condition,
        {"tools": "image_tools", "__end__": "publish"},
    )
    graph.add_edge("image_tools", "image")

    # Publish loop
    graph.add_conditional_edges(
        "publish",
        tools_condition,
        {"tools": "publish_tools", "__end__": END},
    )
    graph.add_edge("publish_tools", END)

    memory = InMemorySaver()
    return graph.compile(checkpointer=memory)


# ====================== LABELS UI ================================

NODE_LABELS = {
    "speech_to_text": ("🎙️", "Transcription audio"),
    "router": ("🧭", "Analyse de la demande"),
    "research": ("🔎", "Recherche web"),
    "research_tools": ("🔎", "Outil Tavily"),
    "writer": ("✍️", "Rédaction de l'histoire"),
    "story_tools": ("✍️", "Génération du texte"),
    "image": ("🎨", "Génération d'image"),
    "image_tools": ("🎨", "Création de l'image"),
    "publish": ("📤", "Publication"),
    "publish_tools": ("📤", "Envoi Zapier"),
}


async def stream_pipeline(
    user_text: Optional[str] = None,
    audio_path: Optional[str] = None,
    thread_id: Optional[str] = None,
) -> AsyncGenerator[dict, None]:
    """Exécute le graphe en streaming.

    Yields:
        {"node": str, "update": dict}
        et un dernier item {"node": "__end__", "update": None, "state": dict}
    """
    builder = build_graph()
    thread_id = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    messages: list[BaseMessage] = []
    if user_text:
        messages.append(HumanMessage(content=user_text))

    input_state: dict = {
        "messages": messages,
        "audio_path": audio_path,
        "research": [],
        "story": [],
        "image_urls": [],
    }

    async for update in builder.astream(
        input_state, config=config, stream_mode="updates"
    ):
        for node_name, node_update in update.items():
            # Enrichir le state.research / story / image_urls à la volée
            if node_name == "research_tools" and isinstance(node_update, dict):
                msgs = node_update.get("messages") or []
                tool_results = [
                    str(m.content)
                    for m in msgs
                    if isinstance(m, ToolMessage) and m.content
                ]
                if tool_results:
                    # On force l'ajout dans le state via un update supplémentaire
                    # (le reducer operator.add s'occupe du merge)
                    yield {
                        "node": node_name,
                        "update": {**node_update, "research": tool_results},
                    }
                    continue
            if node_name == "story_tools" and isinstance(node_update, dict):
                msgs = node_update.get("messages") or []
                for m in msgs:
                    if isinstance(m, ToolMessage) and m.content:
                        node_update = {
                            **node_update,
                            "story": [str(m.content)],
                        }
            if node_name == "image_tools" and isinstance(node_update, dict):
                msgs = node_update.get("messages") or []
                for m in msgs:
                    if isinstance(m, ToolMessage) and m.content:
                        node_update = {
                            **node_update,
                            "image_urls": [str(m.content)],
                        }
            yield {"node": node_name, "update": node_update}

    final_state = await builder.aget_state(config)
    yield {
        "node": "__end__",
        "update": None,
        "state": final_state.values if final_state else {},
    }