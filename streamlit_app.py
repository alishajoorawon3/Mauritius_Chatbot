""" Mauritius Tourism Chatbot
-------------------------
Streamlit interface for the hybrid Mauritius tourism chatbot.
Architecture:
    1. Local TF-IDF knowledge base
    2. Gemini AI fallback
    3. Gemini conversation memory
    4. Streamlit chat history
"""
import streamlit as st
from chatbot_core import TourismChatbot, INTENTS, FALLBACK
import llm_fallback
# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Mauritius Tourism Chatbot",
    page_icon="🏝️",
    layout="centered",
)
# =========================================================
# SESSION STATE
# =========================================================
if "bot" not in st.session_state:
    st.session_state.bot = TourismChatbot()
if "history" not in st.session_state:
    st.session_state.history = []
if "ai_thread_id" not in st.session_state:
    st.session_state.ai_thread_id = None
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None
# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("🌴 Explore Mauritius")
    st.subheader("💡 Suggested questions")
    suggestions = [
        "What are the best beaches in Mauritius?",
        "What are the best places to visit in Mauritius?",
        "What food should I try in Mauritius?",
        "What is the best time to visit Mauritius?",
        "How can I get around Mauritius?",
        "What activities are suitable for families?",
        "Can you create a 7-day itinerary?",
        "What can I do in Mauritius on a budget?",
    ]
    for i, question in enumerate(suggestions):
        if st.button(question, key=f"suggestion_{i}",
                     use_container_width=True):
            st.session_state.pending_question = question
            st.rerun()
    st.subheader("📚 Knowledge base")
    st.caption(f"{len(INTENTS)} tourism topics available "
               "in the local knowledge base.")
    st.subheader("🧠 Conversation memory")
    if st.session_state.ai_thread_id:
        st.success("Active")
    else:
        st.info("Memory becomes active when the AI assistant "
                "handles a conversation.")
    if st.button("🗑️ Start New Conversation",
                 use_container_width=True):
        st.session_state.history = []
        st.session_state.ai_thread_id = None
        st.session_state.pending_question = None
        st.rerun()
# =========================================================
# DISPLAY PREVIOUS MESSAGES
# =========================================================
for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.markdown(message["text"])
        if message["role"] == "assistant":
            source = message.get("source")
            if source == "knowledge_base":
                st.caption("📚 Answered using the Mauritius "
                           "tourism knowledge base.")
            elif source == "ai_assistant":
                st.caption("🤖 Answered by the AI tourism assistant.")
            elif source == "fallback":
                st.caption("ℹ️ General fallback response.")
# =========================================================
# PROCESS USER QUESTION
# =========================================================
user_question = st.chat_input("Ask me anything about Mauritius...") \
    or st.session_state.pending_question
st.session_state.pending_question = None
if user_question:
    user_question = user_question.strip()
    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.history.append(
        {"role": "user", "text": user_question, "source": None}
    )
    reply = None
    source = "fallback"
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # --------- LAYER 1: LOCAL KNOWLEDGE BASE ---------
                debug_result = st.session_state.bot.respond_debug(
                    user_question
                )
                if debug_result.get("is_confident", False):
                    reply = debug_result.get("response", FALLBACK)
                    source = "knowledge_base"
                # --------- LAYER 2: GEMINI AI ---------
                elif llm_fallback.is_available():
                    ai_reply, new_thread_id = llm_fallback.answer(
                        user_question,
                        previous_interaction_id=st.session_state.ai_thread_id,
                    )
                    if ai_reply:
                        reply = ai_reply
                        source = "ai_assistant"
                        if new_thread_id:
                            st.session_state.ai_thread_id = new_thread_id
                    else:
                        reply = ("I'm sorry, I couldn't generate an answer "
                                 "right now. Please try rephrasing your "
                                 "question.")
                        source = "fallback"
                # --------- GEMINI UNAVAILABLE ---------
                else:
                    reply = FALLBACK
                    source = "fallback"
            except Exception as exc:
                print("[streamlit_app] ERROR:",
                      type(exc).__name__, str(exc))
                reply = (
                    "⚠️ **An error occurred while processing "
                    "your question.**\n\n"
                    f"**Error type:** `{type(exc).__name__}`\n\n"
                    f"**Details:** `{str(exc)}`"
                )
                source = "error"
        st.markdown(reply)
        if source == "knowledge_base":
            st.caption("📚 Source: Mauritius tourism knowledge base")
        elif source == "ai_assistant":
            st.caption("🤖 Source: Gemini AI tourism assistant")
        elif source == "fallback":
            st.caption("ℹ️ General fallback response")
    st.session_state.history.append(
        {"role": "assistant", "text": reply, "source": source}
    )
