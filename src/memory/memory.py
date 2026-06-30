"""
src.memory.memory — Gestion de la mémoire court et long terme de l'agent
"""

from langgraph.checkpoint.memory import MemorySaver


def get_memory() -> MemorySaver:
    """
    Retourne un checkpointer mémoire en RAM (court terme).
    Pour une mémoire persistante, remplacer par SqliteSaver ou PostgresSaver.
    """
    return MemorySaver()
