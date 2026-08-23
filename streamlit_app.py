"""
Mauritius Tourism Chatbot - v2.
Deployable on Streamlit Community Cloud: needs this file, chatbot_core.py,
llm_fallback.py, and requirements.txt in the same repo/folder.

Two layers:
  1. Offline TF-IDF intent matcher (chatbot_core.py) - instant, free,
     answers confidently-recognised questions from a curated knowledge base
     of 34 travel topics.
  2. Claude fallback (llm_fallback.py) - only used when layer 1 isn't
     confident, so genuinely open-ended or novel questions still get a
     real answer instead of "please rephrase". Requires an ANTHROPIC_API_KEY
     Streamlit secret; the app still works with layer 1 alone if it's absent.

See SETUP.md for how to add the API key.
"""

import streamlit as st
from chatbot_core import TourismChatbot, INTENTS, FALLBACK
import llm_fallback

st.set_page_config(page_title="Mauritius Tourism Chatbot", page_icon="🏝️")
st.title("🏝️ Mauritius Tourism Chatbot")
st.caption("Ask me anything about visiting Mauritius - visas, safety, budget, "
           "food, weather, honeymoons, diving, and more.")

if "bot" not in st.session_state:
    st.session_state.bot = TourismChatbot()
if "history" not in st.session_state:
    st.session_state.history = []
if "stats" not in st.session_state:
    st.session_state.stats = {"knowledge_base": 0, "ai_assistant": 0, "unanswered": 0}

llm_on = llm_fallback.is_available()

prompt = st.chat_input("Ask me about visiting Mauritius...")
if prompt:
    dbg = st.session_state.bot.respond_debug(prompt)
    if dbg["is_confident"]:
        reply = dbg["response"]
        source = "knowledge_base"
    elif llm_on:
        llm_reply = llm_fallback.answer(prompt)
        if llm_reply:
            reply = llm_reply
            source = "ai_assistant"
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
    if not llm_on:
        st.warning("AI fallback is off - no ANTHROPIC_API_KEY configured. "
                    "See SETUP.md to enable it.", icon="⚠️")
    st.caption("Prototype built for a BSc dissertation on tourism analytics "
               "and conversational AI for Mauritius.")
