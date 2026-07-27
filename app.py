import streamlit as st
import asyncio
import uuid
from audio_recorder_streamlit import audio_recorder
from langchain_core.messages import HumanMessage
from main import builder

# Configuration de la page
st.set_page_config(page_title="AI Creator Studio", page_icon="🚀", layout="centered")

st.title("🚀 AI Content Factory")
st.markdown("Générez des histoires ou des images à partir de texte ou de votre voix.")

# Initialisation de la session
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Sidebar : Configuration
with st.sidebar:
    st.header("Configuration")
    auto_publish = st.checkbox("Publication automatique sur LinkedIn", value=False)
    st.info("Les agents traiteront votre demande automatiquement.")

# Interface : Entrée (Texte ou Micro)
tab1, tab2 = st.tabs(["✍️ Texte ou Image", "🎙️ Micro"])

user_input = None

with tab1:
    text_input = st.text_area("Décrivez votre besoin :", height=100)
    if st.button("Lancer la génération (Texte)"):
        user_input = text_input

with tab2:
    audio_bytes = audio_recorder(text="Cliquez pour parler", icon_size="2x")
    if audio_bytes:
        with open("temp_input.wav", "wb") as f:
            f.write(audio_bytes)
        # Note: Le traitement audio se fera dans le main.py via le nœud speech_to_text
        user_input = "AUDIO_FILE" 

# --- Logique de Streaming ---
if user_input:
    st.divider()
    container = st.empty()
    full_response = ""
    
# ==== Configuration du graphe
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    input_data = {
        "messages": [HumanMessage(content=user_input if user_input != "AUDIO_FILE" else "")],
        "audio_path": "temp_input.wav" if user_input == "AUDIO_FILE" else None
    }

    async def stream_output():
        full_text = ""
        # On utilise astream_events pour capturer tout ce qui se passe
        async for event in builder.astream_events(input_data, config=config, version="v1"):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"].content
                if chunk:
                    full_text += chunk
                    yield full_text

    # Affichage du streaming
    async def run():
        response_box = st.empty()
        async for text in stream_output():
            response_box.markdown(text + "▌")
        response_box.markdown(text)

    asyncio.run(run())
    st.success("Tâche terminée !")

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Statut: Prêt à générer")