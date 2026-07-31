# Narrator AI Agent

## Description
**Narrator AI Agent** est une plateforme d'automatisation de création de contenu multimodal construite avec **LangGraph** et **Streamlit**. L'agent analyse les besoins de l'utilisateur, effectue des recherches sur le web, génère des récits créatifs ou des images, et publie le contenu automatiquement via des webhooks (Zapier).

## Architecture Technique

Le système repose sur un graphe d'états (StateGraph) géré par LangGraph, permettant une orchestration complexe et asynchrone des agents :
*   **Routage intelligent** : Analyse la nature de la demande (histoire vs image).
*   **Recherche Web** : Utilise Tavily pour enrichir le contenu avec des données réelles.
*   **Génération Multimodale** : Utilise les LLM pour la rédaction et les modèles spécialisés (Pollinations) pour la génération d'images.
*   **Publication** : Intégration Zapier pour la diffusion automatisée.

## Structure du Projet

- `agent.py` : Logique métier, définition du graphe LangGraph et des outils.
- `ui.py` : Interface utilisateur Streamlit avancée et gestion de l'état.
- `app.py` : Point d'entrée simple pour des tests rapides.
- `pyproject.toml` : Gestion des dépendances du projet.

## Installation

Assurez-vous d'avoir Python 3.10+ installé.

```bash

pip install -r requirements.txt
```

## Configuration

Créez un fichier `.env` à la racine et configurez les clés API suivantes :
- `OPENAI_API_KEY`
- `HF_API_TOKEN`
- `GOOGLE_API_KEY`
- `TAVILY_API_KEY`
- `ZAPIER_WEBHOOK_URL` (optionnel)

## Utilisation

Lancez l'interface principale :
```bash
streamlit run ui.py
```
