"""
Content Studio AI — Interface Streamlit
Pipeline multimodal LangGraph : routage -> recherche -> rédaction / image -> publication
"""

import asyncio
import tempfile
import uuid
from datetime import datetime

import streamlit as st

from agent_pipeline import stream_pipeline, check_env, NODE_LABELS

# ----------------------------------------------------------------------------
# CONFIG PAGE
# ----------------------------------------------------------------------------

st.set_page_config(
    page_title="NARRATOR AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# THEME / CSS
# ----------------------------------------------------------------------------

PRIMARY = "#4F46E5"       # indigo
PRIMARY_DARK = "#3730A3"
ACCENT = "#14B8A6"        # teal
BG = "#0F172A"            # slate 900
CARD_BG = "#1E293B"       # slate 800
TEXT = "#E2E8F0"
MUTED = "#94A3B8"
SUCCESS = "#22C55E"
WARNING = "#F59E0B"

st.markdown(
    f"""
    <style>
        .stApp {{
            background: radial-gradient(circle at 20% 0%, #1E1B4B 0%, {BG} 45%);
            color: {TEXT};
        }}
        section[data-testid="stSidebar"] {{
            background-color: {CARD_BG};
            border-right: 1px solid rgba(255,255,255,0.06);
        }}
        h1, h2, h3, h4 {{
            font-family: 'Inter', sans-serif;
            letter-spacing: -0.02em;
        }}
        .studio-title {{
            font-size: 1.6rem;
            font-weight: 800;
            background: linear-gradient(90deg, {PRIMARY} 0%, {ACCENT} 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0;
        }}
        .studio-subtitle {{
            color: {MUTED};
            font-size: 0.85rem;
            margin-top: -6px;
        }}
        .card {{
            background: {CARD_BG};
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 16px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1rem;
        }}
        .badge {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 600;
            margin-right: 6px;
        }}
        .badge-ok {{ background: rgba(34,197,94,0.15); color: {SUCCESS}; border: 1px solid rgba(34,197,94,0.35); }}
        .badge-ko {{ background: rgba(245,158,11,0.15); color: {WARNING}; border: 1px solid rgba(245,158,11,0.35); }}
        .badge-type {{ background: rgba(79,70,229,0.18); color: #A5B4FC; border: 1px solid rgba(79,70,229,0.4); }}

        div.stButton > button {{
            background: linear-gradient(90deg, {PRIMARY} 0%, {PRIMARY_DARK} 100%);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.55rem 1.4rem;
            font-weight: 600;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }}
        div.stButton > button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 6px 18px rgba(79,70,229,0.35);
        }}
        div.stButton > button:focus {{
            box-shadow: 0 0 0 2px {ACCENT};
        }}
        .secondary-btn button {{
            background: transparent !important;
            border: 1px solid rgba(255,255,255,0.15) !important;
            color: {TEXT} !important;
        }}
        [data-testid="stChatMessage"] {{
            background: {CARD_BG};
            border-radius: 14px;
            border: 1px solid rgba(255,255,255,0.06);
        }}
        hr {{
            border-color: rgba(255,255,255,0.08);
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------------------------

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: {role, content, ts, type}
if "running" not in st.session_state:
    st.session_state.running = False

# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------

with st.sidebar:
    st.markdown('<p class="studio-title">🎬 Content Studio AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="studio-subtitle">Agent multimodal · LangGraph</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**Statut des clés API**")
    env = check_env()
    labels = {
        "GOOGLE_API_KEY": "Gemini (routage / image)",
        "HF_API_TOKEN": "HuggingFace (Qwen)",
        "TAVILY_API_KEY": "Tavily (recherche web)",
        "OPENAI_API_KEY": "OpenAI (transcription)",
    }
    for key, label in labels.items():
        ok = env.get(key, False)
        cls = "badge-ok" if ok else "badge-ko"
        icon = "✓" if ok else "✗"
        st.markdown(
            f'<span class="badge {cls}">{icon} {label}</span>', unsafe_allow_html=True
        )

    st.markdown("---")
    st.caption(f"Session : `{st.session_state.thread_id[:8]}`")
    if st.button("🔄 Nouvelle conversation", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.history = []
        st.rerun()

    st.markdown("---")
    st.caption(
        "Pipeline : Transcription → Routage → Recherche (optionnelle) "
        "→ Rédaction / Image → Publication"
    )

# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------

st.markdown(
    """
    <h2 style="margin-bottom:0;">Studio de génération de contenu</h2>
    <p style="color:#94A3B8; margin-top:4px;">
        Décris ce que tu veux produire — une histoire ou une image — par texte ou à l'oral.
        L'agent choisit lui-même s'il doit faire une recherche web avant de créer, puis publie le résultat.
    </p>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# INPUT PANEL
# ----------------------------------------------------------------------------

col_in, col_out = st.columns([1, 1.3], gap="large")

with col_in:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 📝 Entrée")

    tab_text, tab_audio = st.tabs(["Texte", "🎙️ Micro"])

    user_text = None
    audio_path = None

    with tab_text:
        user_text = st.text_area(
            "Ta demande",
            placeholder="Ex : Écris une histoire de science-fiction sur une IA qui explore Mars, "
                        "en te basant sur les dernières missions martiennes réelles.",
            height=140,
            label_visibility="collapsed",
        )

    with tab_audio:
        audio_value = st.audio_input("Enregistre ta demande")
        if audio_value is not None:
            st.audio(audio_value)

    col_a, col_b = st.columns([1, 1])
    with col_a:
        launch = st.button("🚀 Générer", use_container_width=True, disabled=st.session_state.running)
    with col_b:
        st.markdown('<div class="secondary-btn">', unsafe_allow_html=True)
        clear = st.button("🗑️ Effacer l'historique", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

if clear:
    st.session_state.history = []
    st.rerun()

# ----------------------------------------------------------------------------
# EXECUTION
# ----------------------------------------------------------------------------

with col_out:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### ⚙️ Suivi du pipeline")
    status_container = st.container()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 📦 Résultat")
    result_container = st.container()
    st.markdown('</div>', unsafe_allow_html=True)


def save_audio_to_tempfile(audio_value) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.write(audio_value.getvalue())
    tmp.flush()
    return tmp.name


async def run_pipeline_ui(user_text, audio_path):
    status_container.empty()
    result_container.empty()

    story_text, image_urls, publish_result, content_type = None, None, None, None

    with status_container:
        step_placeholder = st.empty()
        log_lines = []

        async for event in stream_pipeline(user_text, audio_path, st.session_state.thread_id):
            node = event["node"]

            if node == "__end__":
                state = event["state"] or {}
                story_list = state.get("story") or []
                image_list = state.get("image_urls") or []
                story_text = story_list[-1] if story_list else None
                image_urls = image_list[-1] if image_list else None
                publish_result = state.get("publish_result")
                content_type = state.get("content_type")
                break

            icon, label = NODE_LABELS.get(node, ("⚙️", node))
            log_lines.append(f"{icon} **{label}** — terminé")
            step_placeholder.markdown("\n\n".join(log_lines))

    with result_container:
        if content_type:
            type_label = "📖 Histoire" if content_type == "story" else "🖼️ Image"
            st.markdown(f'<span class="badge badge-type">{type_label}</span>', unsafe_allow_html=True)
            st.write("")

        if story_text:
            st.markdown(story_text)
        if image_urls:
            st.info(f"Image générée (URL simulée) : {image_urls}")

        if publish_result:
            st.markdown(
                f'<span class="badge badge-ok">✓ Publié avec succès</span>',
                unsafe_allow_html=True,
            )

    st.session_state.history.append(
        {
            "role": "user",
            "content": user_text or "🎙️ (message audio)",
            "ts": datetime.now().strftime("%H:%M"),
        }
    )
    st.session_state.history.append(
        {
            "role": "assistant",
            "content": story_text or (f"Image générée : {image_urls}" if image_urls else "Terminé."),
            "ts": datetime.now().strftime("%H:%M"),
        }
    )


if launch:
    if not user_text and audio_value is None:
        st.warning("Ajoute un texte ou un enregistrement audio avant de lancer la génération.")
    else:
        st.session_state.running = True
        path = save_audio_to_tempfile(audio_value) if audio_value is not None else None
        try:
            asyncio.run(run_pipeline_ui(user_text.strip() if user_text else None, path))
        except Exception as exc:
            st.error(f"Erreur pendant l'exécution du pipeline : {exc}")
        finally:
            st.session_state.running = False

# ----------------------------------------------------------------------------
# HISTORIQUE
# ----------------------------------------------------------------------------

if st.session_state.history:
    st.markdown("---")
    st.markdown("#### 💬 Historique de la conversation")
    for msg in st.session_state.history:
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.markdown(msg["content"])
            st.caption(msg["ts"])