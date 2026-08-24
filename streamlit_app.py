"""
Streamlit web interface for the tourism chatbot - branded "Mauritius Chatbot".
Deployable on Streamlit Community Cloud: needs this file, chatbot_core.py,
llm_fallback.py, and requirements.txt in the same repo/folder.

Two layers:
  1. Offline TF-IDF intent matcher (chatbot_core.py) - instant, free,
     answers confidently-recognised questions from a curated knowledge base
     of 40 travel/country topics.
  2. Gemini fallback (llm_fallback.py) - only used when layer 1 isn't
     confident, so genuinely open-ended or novel questions still get a
     real answer instead of "please rephrase". Requires a GEMINI_API_KEY
     Streamlit secret; the app still works with layer 1 alone if it's absent.

Conversation memory: st.session_state.ai_thread_id tracks the most recent
Gemini interaction id so follow-up questions ("what about for families?")
are understood in context instead of answered from a blank slate each time.
Only AI-answered turns extend the thread - a question caught by the offline
matcher in between doesn't get added to Gemini's memory of the conversation,
since it never talks to Gemini at all (see llm_fallback.py's docstring).

See SETUP.md for how to add the API key.
"""

import streamlit as st
from chatbot_core import TourismChatbot, INTENTS, FALLBACK
import llm_fallback

st.set_page_config(page_title="Mauritius Chatbot", page_icon="🏝️")
st.title("🏝️ Mauritius Chatbot")
st.caption("Ask me anything about visiting Mauritius - visas, safety, budget, "
           "food, weather, honeymoons, diving, history, and more.")

if "bot" not in st.session_state:
    st.session_state.bot = TourismChatbot()
if "history" not in st.session_state:
    st.session_state.history = []
if "stats" not in st.session_state:
    st.session_state.stats = {"knowledge_base": 0, "ai_assistant": 0, "unanswered": 0}
if "ai_thread_id" not in st.session_state:
    st.session_state.ai_thread_id = None

llm_on = llm_fallback.is_available()

prompt = st.chat_input("Ask me about visiting Mauritius...")
if prompt:
    dbg = st.session_state.bot.respond_debug(prompt)
    if dbg["is_confident"]:
        reply = dbg["response"]
        source = "knowledge_base"
    elif llm_on:
        llm_reply, new_thread_id = llm_fallback.answer(
            prompt, previous_interaction_id=st.session_state.ai_thread_id
        )
        if llm_reply:
            reply = llm_reply
            source = "ai_assistant"
            st.session_state.ai_thread_id = new_thread_id
        else:
            reply = FALLBACK
            source = "unanswered"
    else:
        reply = FALLBACK
        source = "unanswered"

    st.session_state.stats[source] += 1
    st.session_state.history.append(("user", prompt, None))
    st.session_state.history.append(("assistant", reply, source))

for role, text, source in st.session_state.history:
    with st.chat_message(role):
        st.write(text)
        if source == "ai_assistant":
            st.caption("🤖 answered by Gemini (topic outside the built-in knowledge base)")

with st.sidebar:
    st.subheader("Try asking about:")
    topic_names = [name.replace("_", " ").title() for name in INTENTS]
    st.markdown(", ".join(topic_names))
    st.caption("...plus genuinely open-ended questions, handled by an AI "
               "fallback." if llm_on else
               "...plus other questions, though without an AI fallback "
               "configured, anything outside these topics gets a generic "
               "message (see SETUP.md).")

    st.divider()
    st.subheader("This session")
    total = sum(st.session_state.stats.values())
    if total:
        st.metric("Answered from knowledge base", st.session_state.stats["knowledge_base"])
        st.metric("Answered by AI fallback", st.session_state.stats["ai_assistant"])
        st.metric("Unanswered", st.session_state.stats["unanswered"])
    else:
        st.caption("Ask a question to see stats here.")

    st.divider()
    if st.button("🔄 Start a new conversation"):
        st.session_state.history = []
        st.session_state.ai_thread_id = None
        st.session_state.stats = {"knowledge_base": 0, "ai_assistant": 0, "unanswered": 0}
        st.rerun()
    st.caption("Clears the chat and the AI's memory of what you've discussed "
               "so far - use this if you want to switch topics with a clean "
               "slate.")

    st.divider()
    if not llm_on:
        st.warning("AI fallback is off - no GEMINI_API_KEY configured. "
                    "See SETUP.md to enable it.", icon="⚠️")
    st.caption("Prototype built for a BSc dissertation on tourism analytics "
               "and conversational AI for Mauritius.")
