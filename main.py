# ===================== LES DEPENDANCES ==============
# Configs

from typing import List, Annotated, Literal, Optional, TypedDict
from langgraph.graph import StateGraph, START, END

from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, BaseMessage
# from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
import os
import httpx
import asyncio
from openai import AsyncOpenAI

# Client Tavily ASYNC (et non le client synchrone TavilyClient)
from tavily import AsyncTavilyClient

load_dotenv()

os.getenv("OPENAI_API_KEY")
os.getenv("TAVILY_API_KEY")
os.getenv("HF_API_TOKEN")
os.getenv("GOOGLE_API_KEY")

# ============== DEFINIR LES LLMs ===============

from huggingface_hub import login

login()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    max_new_tokens=256,
    temperature=0.7,
    repetition_penalty=1.3,
    do_sample=True
)

ibm_llm=ChatHuggingFace(llm=llm)

goog_llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
    streaming=True,
    repetition_penalty=1.3,
    do_sample=True
)

# ====================== DEFINIR LES TOOLS ==========================

client = AsyncTavilyClient()  # client async, cohérent avec les tools async


@tool
async def research_web(query: str) -> str:
    """
    Recherche des informations fiables sur le web.

    Args:
        query: La requête de recherche de l'utilisateur.

    Returns:
        str: Résultats de recherche sous forme de texte.
    """
    response = await client.search(
        query=query,
        search_depth="advanced",
        max_results=5
    )

    results = response.get("results", [])

    return "\n\n".join(
        f"- {r['title']}\n  {r['url']}\n  {r['content']}"
        for r in results
    )


@tool
async def recit_writer(
    subject: str,
    style: str = "Narratif",
    language: str = "Français",
    length: str = "moyen"
) -> str:
    """
    Génère une histoire ou un récit de qualité.

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

    response = await ibm_llm.ainvoke(prompt)

    return response.content


@tool
async def image_generate(prompt: str, style: Optional[str] = None) -> str:
    """
    Génère une image à l'aide de Gemini.

    Args:
        prompt: Description détaillée de l'image à générer.
        style: Style artistique optionnel (cinématographique, illustration, réaliste...).

    Returns:
        Chemin ou URL de l'image générée.
    """
    if style:
        full_prompt = f"{prompt}, style {style}, haute qualité, détaillé"
    else:
        full_prompt = prompt

    return f"https://gemini-generated-image/{hash(full_prompt)}.png"


@tool
async def publish_content(
    text: Optional[str] = None,
    image_urls: Optional[list[str]] = None
):
    """
    Publie un contenu texte, une image,
    ou un contenu multimédia via Zapier.
    """
    ZAPIER_WEBHOOK_URL = "https://hooks.zapier.com/hooks/catch/28063276/428jcbk/"
    payload = {}

    if text:
        payload["text"] = text.strip()

    if image_urls:
        payload["image"] = image_urls

    async with httpx.AsyncClient() as client:
        response = await client.post(
            ZAPIER_WEBHOOK_URL,
            json=payload
        )
        response.raise_for_status()

        return response.json()


# ==================== L'ETAT DU GRAPH ==================

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class Route(BaseModel):
    content_type: Literal["story", "image"] = Field(
        description="Le type de contenu final à produire : une histoire ou une image."
    )
    needs_research: bool = Field(
        description="True si une recherche d'informations externes est nécessaire avant de produire le contenu."
    )


class State(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages]

    audio_path:Optional[str]
    

    content_type: Literal["story", "image"]
    needs_research: bool

    research: list[str]
    story: list[str] | None
    image_urls: Optional[list[str]]
    publish_result: dict | None
   

# =============== CREER LES SYSTEMES_PROMPTS ==============

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

Ton rôle est de trouver des informations fiables et pertinentes en utilisant
l'outil de recherche web Tavily lorsque cela est nécessaire.

Tes responsabilités :

1. Analyse la demande de l'utilisateur avant de répondre.
2. Si la réponse nécessite des informations externes, récentes ou vérifiables,
   utilise obligatoirement l'outil Tavily.
3. Formule des requêtes de recherche précises et adaptées au besoin.
4. Analyse les résultats retournés par Tavily et sélectionne uniquement les
   informations pertinentes et fiables.
5. Synthétise les informations trouvées de manière claire et structurée.
6. Mentionne les sources ou références disponibles lorsque cela est utile.

Règles importantes :

- Ne fabrique jamais d'informations qui ne proviennent pas des sources
  disponibles ou de tes connaissances fiables.
- Si les résultats de recherche sont insuffisants, indique clairement
  que les informations trouvées sont limitées.
- Ne réponds pas uniquement avec une liste de liens : explique les résultats.
- Adapte la profondeur de la réponse à la demande de l'utilisateur.

Tu disposes d'un outil de recherche web appelé `research_web`.
Utilise cet outil dès qu'une recherche externe est nécessaire.
"""

STORY_SYSTEM_PROMPT = """
Tu es un agent spécialisé dans la création d'histoires et de récits narratifs.

Ton rôle est de transformer des informations, des idées ou des concepts fournis
par l'utilisateur ou par d'autres agents en une histoire captivante, structurée
et agréable à lire.

Tes responsabilités :

1. Comprendre l'objectif de l'histoire :
   - divertissement ;
   - éducation ;
   - storytelling professionnel ;
   - sensibilisation ;
   - narration pour réseaux sociaux.

2. Construire une histoire avec :
   - une introduction qui attire l'attention ;
   - des personnages ou éléments principaux clairement définis ;
   - un développement logique ;
   - des événements cohérents ;
   - une conclusion mémorable.

3. Adapter le style narratif au contexte demandé :
   - conte ;
   - récit réaliste ;
   - science-fiction ;
   - aventure ;
   - histoire éducative ;
   - storytelling marketing.

4. Créer des descriptions riches et immersives :
   - émotions ;
   - ambiance ;
   - détails visuels ;
   - dialogues lorsque cela améliore le récit.

Règles importantes :

- Ne crée jamais de faits présentés comme vrais si aucune information fiable
  n'est fournie.
- Si l'histoire utilise des informations provenant d'un agent de recherche,
  respecte les faits importants tout en les transformant en narration.
- Privilégie une structure claire et une progression naturelle.
- Évite les répétitions et les formulations trop génériques.
- Adapte la longueur de l'histoire à la demande de l'utilisateur.
- Utilise obligatoirement l'outil `recit_writer` pour générer le texte final de l'histoire.

Tu es un écrivain professionnel capable de transformer une simple idée en
une histoire engageante et mémorable.
"""

IMAGE_SYSTEM_PROMPT = """
Tu es un agent spécialisé dans la création et la génération d'images.

Ton rôle est de transformer les demandes des utilisateurs en descriptions
visuelles détaillées et d'utiliser l'outil `image_generate` pour créer des
images adaptées à leurs besoins.

Tes responsabilités :

1. Analyser la demande de l'utilisateur :
   - identifier le sujet principal ;
   - comprendre le contexte ;
   - déterminer l'objectif de l'image ;
   - identifier le public cible si nécessaire.

2. Avant d'appeler l'outil `image_generate`, améliorer la description de l'image
   en précisant :
   - le sujet principal ;
   - la composition de la scène ;
   - l'environnement ;
   - les couleurs et l'ambiance ;
   - le niveau de détail ;
   - le style artistique approprié.

3. Choisir un style adapté lorsque cela améliore le résultat :
   - réaliste ;
   - cinématographique ;
   - illustration ;
   - dessin animé ;
   - peinture numérique ;
   - style futuriste ;
   - photographie professionnelle.

4. Utiliser obligatoirement l'outil `image_generate` lorsque l'utilisateur
   demande :
   - une image ;
   - une illustration ;
   - une représentation visuelle ;
   - une couverture ;
   - une scène ou un personnage.

5. Après la génération :
   - retourner clairement le résultat obtenu ;
   - expliquer brièvement l'image créée ;
   - fournir l'URL ou le chemin retourné par l'outil.

Règles importantes :

- Ne génère jamais une image uniquement avec ton imagination :
  utilise toujours l'outil `image_generate`.
- Ne modifie pas l'intention originale de l'utilisateur.
- Si la demande est trop vague, demande des précisions sur le sujet,
  le style ou l'ambiance avant la génération.
- Crée des prompts d'image précis et riches pour obtenir une meilleure qualité.
- Évite les descriptions ambiguës.

Tu es un expert en prompt engineering pour la génération d'images.
Ton objectif est de transformer une idée textuelle en une représentation
visuelle de haute qualité.
"""

PUBLISHER_SYSTEM_PROMPT = """
Tu es un agent spécialisé dans la publication de contenus numériques.

Ton rôle est de publier des contenus finalisés (histoires, textes, images ou
contenus multimédias) vers des plateformes externes en utilisant l'outil
`publish_content`.

Tu ne crées pas de contenu.
Tu interviens uniquement lorsque le contenu est prêt à être publié.

Tes responsabilités :

1. Analyser le contenu disponible dans la conversation :
   - identifier s'il existe un texte à publier ;
   - identifier s'il existe une image à publier ;
   - déterminer si le contenu est textuel, visuel ou multimédia.

2. Utiliser l'outil `publish_content` selon le besoin :

   - Pour une histoire ou un texte uniquement :
     appeler :
        publish_content(text="...")

   - Pour une image uniquement :
     appeler :
        publish_content(image_urls=["..."])

   - Pour une histoire accompagnée d'une image :
     appeler :
        publish_content(
            text="...",
            image_urls=["..."]
        )

3. Vérifier avant publication que :
   - le contenu n'est pas vide ;
   - le texte est correctement formaté ;
   - l'URL de l'image est valide si une image est fournie.

4. Après publication :
   - confirmer que l'envoi a été effectué ;
   - retourner le statut ou la réponse obtenue depuis l'outil.

Règles importantes :

- Ne génère jamais d'histoire ou d'image toi-même.
- Ne modifie pas le contenu original sauf pour améliorer légèrement
  la mise en forme si nécessaire.
- N'appelle pas l'outil `publish_content` si aucun contenu publiable
  n'est disponible.
- Si des informations manquent (texte ou image attendu mais absent),
  demande ou signale ce qui manque avant de publier.
- Respecte toujours le choix de l'utilisateur :
  une histoire seule, une image seule ou les deux.

Tu es le dernier maillon de la chaîne de production.
Ta mission est de garantir que le bon contenu est envoyé au bon moment
vers le système de publication.
"""

# ================= LLMs AVEC TOOLS ================

llm_search = ibm_llm.bind_tools(tools=[research_web])
llm_image = goog_llm.bind_tools(tools=[image_generate])
llm_story = ibm_llm.bind_tools(tools=[recit_writer])
llm_publish = goog_llm.bind_tools(tools=[publish_content])


# ==============================================================================
#                              CREER LES NODES 
# ===============================================================================

# =============== Transcription Node ===================

openai_client = AsyncOpenAI()


async def speech_to_text(state:State) -> dict[str, list[BaseMessage]]:
    """
    Transcrit un fichier audio en texte à l'aide d'OpenAI.

    Args:
        state: état courant du graphe contenant le chemin du fichier audio.

    Returns:
        Mise à jour de l'état avec le message transcrit.
    """
    audio_path = state.get("audio_path")
    if not audio_path:
        return {}
    with open(state["audio_path"], "rb") as audio:
        transcription = await openai_client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=audio
        )
    return {
        "messages": [
            HumanMessage(content=transcription.text)
        ]
    }

# ================ Router Node =================================


router_llm = goog_llm.with_structured_output(Route)


async def router_agent(state: State) -> dict:
    decision: Route = await router_llm.ainvoke(
        [
            SystemMessage(content=ROUTER_PROMPT),
            *state["messages"]
        ]
    )
    return {
        "content_type": decision.content_type,
        "needs_research": decision.needs_research,
    }


# ================== Node1 : Research ==========================


async def research_agent(state: State) -> dict:
    messages = [SystemMessage(content=RESEARCH_SYSTEM_PROMPT), *state["messages"]]
    response = await llm_search.ainvoke(messages)
    return {"messages": [response], "research": [response.content]}


# ================== Node2 : Writer ==========================

async def writer_agent(state: State) -> dict:
    messages = [SystemMessage(content=STORY_SYSTEM_PROMPT), *state["messages"]]
    response = await llm_story.ainvoke(messages)
    return {"messages": [response], "story": [response.content]}


# ================== Node3 : Image =============================


async def image_agent(state: State) -> dict:
    messages = [SystemMessage(content=IMAGE_SYSTEM_PROMPT), *state["messages"]]
    response = await llm_image.ainvoke(messages)
    return {"messages": [response], "image_urls": [response.content]}


# ================== Node4 : Publish ============================


async def publish_agent(state: State) -> dict:
    messages = [SystemMessage(content=PUBLISHER_SYSTEM_PROMPT), *state["messages"]]
    story_list = state.get("story") or []
    image_list = state.get("image_urls") or []
    content = " ".join(story_list) if story_list else (image_list[0] if image_list else "")

    
    if content:
        messages.append(HumanMessage(content=f"Voici le contenu à publier : {content}"))

    response = await llm_publish.ainvoke(messages)

    return {
        "messages": [response],
        "publish_result": {"status": "succes"}
    }


# ============== CONSTRUIRE LE GRAPH ============================


from langgraph.prebuilt import ToolNode, tools_condition

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

graph.add_edge(
    "speech_to_text",
    "router"
)

def route_after_router(state: State) -> str:
    if state["needs_research"]:
        return "research"
    return "writer" if state["content_type"] == "story" else "image"


graph.add_conditional_edges(
    "router",
    route_after_router,
    {"research": "research", "writer": "writer", "image": "image"},
)


def route_from_research(state: State) -> str:
    if tools_condition(state) == "tools":
        return "research_tools"
    return "writer" if state["content_type"] == "story" else "image"


graph.add_conditional_edges(
    "research",
    route_from_research,
    {"tools": "research_tools", "writer": "writer", "image": "image"},
)
graph.add_edge("research_tools", "research")

# --- Writer : boucle ReAct avec recit_writer ---
graph.add_conditional_edges(
    "writer",
    tools_condition,
    {"tools": "story_tools", END: "publish"},
)
graph.add_edge("story_tools", "writer")

# --- Image : boucle ReAct avec image_generate ---
graph.add_conditional_edges(
    "image",
    tools_condition,
    {"tools": "image_tools", END: "publish"},
)
graph.add_edge("image_tools", "image")

# --- Publish : boucle ReAct avec publish_content ---
graph.add_conditional_edges(
    "publish",
    tools_condition,
    {"tools": "publish_tools", END: END},
)
graph.add_edge("publish_tools", "publish")


# ============== EXECUTION =====================

import uuid
from langgraph.checkpoint.memory import InMemorySaver

memory = InMemorySaver()

config = {
    "configurable": {
        "thread_id": str(uuid.uuid4())
    }
}

builder = graph.compile(checkpointer=memory)


async def main():
    resp = await builder.ainvoke(
        input={
            "audio_path": "mon_audio.wav",
            "messages":[]
        },
        config=config
    )
    print(resp["messages"][-1].content)

# =================  EXECUTION =================

if __name__ == "__main__":
    asyncio.run(main())