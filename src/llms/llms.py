"""
src.llms.llms — Configuration et instanciation des LLMs
"""

import os
from dotenv import load_dotenv

load_dotenv()


def get_llm(provider: str = "watsonx"):
    """
    Retourne le LLM configuré selon le fournisseur.

    Fournisseurs supportés : 'watsonx' | 'google' | 'huggingface'
    """
    if provider == "watsonx":
        from langchain_ibm import WatsonxLLM

        return WatsonxLLM(
            model_id=os.getenv("WATSONX_MODEL_ID", "ibm/granite-3-3-8b-instruct"),
            url=os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"),
            apikey=os.getenv("WATSONX_APIKEY", ""),
            project_id=os.getenv("WATSONX_PROJECT_ID", ""),
            params={
                "max_new_tokens": 1024,
                "temperature": 0.7,
            },
        )

    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
            google_api_key=os.getenv("GOOGLE_API_KEY", ""),
            temperature=0.7,
        )

    elif provider == "huggingface":
        from langchain_huggingface import HuggingFaceEndpoint

        return HuggingFaceEndpoint(
            repo_id=os.getenv("HF_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.3"),
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN", ""),
            temperature=0.7,
        )

    else:
        raise ValueError(f"Fournisseur LLM inconnu : '{provider}'. Choisir parmi : watsonx, google, huggingface")
