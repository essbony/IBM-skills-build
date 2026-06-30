"""
src.tools.tools — Outils disponibles pour les agents IA
"""

from langchain_core.tools import tool


@tool
def addition(a: float, b: float) -> float:
    """Additionne deux nombres."""
    return a + b


@tool
def recherche_web(query: str) -> str:
    """Effectue une recherche web et retourne les résultats (simulé)."""
    # TODO: Brancher un vrai moteur (ex: Tavily, SerpAPI)
    return f"[Résultat simulé pour : {query}]"


def get_tools() -> list:
    """Retourne la liste des outils disponibles pour l'agent."""
    return [addition, recherche_web]
