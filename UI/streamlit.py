"""
UI/streamlit.py — Interface conversationnelle avec Streamlit
Pour lancer : streamlit run UI/streamlit.py
"""

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from src.agents import build_agent


# ─── Configuration de la page ────────────────────────────────────────────────
st.set_page_config(page_title="Agent IA", page_icon="🤖", layout="centered")
st.title("🤖 Agent IA — project-bob")
st.caption("Propulsé par LangGraph + IBM WatsonX")

# ─── Initialisation de la session ────────────────────────────────────────────
if "agent" not in st.session_state:
    st.session_state.agent = build_agent()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "streamlit-session-1"

# ─── Affichage de l'historique ───────────────────────────────────────────────
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# ─── Saisie utilisateur ──────────────────────────────────────────────────────
if prompt := st.chat_input("Votre message..."):
    user_msg = HumanMessage(content=prompt)
    st.session_state.messages.append(user_msg)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Réflexion en cours..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            response = st.session_state.agent.invoke(
                {"messages": [user_msg]},
                config=config,
            )
            answer = response["messages"][-1].content

        st.markdown(answer)
        st.session_state.messages.append(AIMessage(content=answer))
