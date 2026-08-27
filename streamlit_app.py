"""
Mauritius Tourism Chatbot
-------------------------
Streamlit interface for the hybrid Mauritius tourism chatbot.

Architecture:
1. Local TF-IDF knowledge base
   - Fast responses for recognised tourism topics.
2. Gemini AI fallback
   - Handles broader and more conversational questions.
3. Conversation memory
   - Maintains Gemini interaction context.
4. Chat history
   - Displays the complete conversation during the current session.

Required files in the same GitHub repository:
- streamlit_app.py
- chatbot_prototype.py
- llm_fallback.py
- requirements.txt

Required Streamlit Secret:
- GEMINI_API_KEY
"""

import streamlit as st

from chatbot_prototype import TourismChatbot, INTENTS, FALLBACK
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
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.4rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: #666666;
            font-size: 1rem;
            margin-bottom: 1.5rem;
        }

        .source-label {
            font-size: 0.75rem;
            color: #777777;
            margin-top: 0.25rem;
        }

        .welcome-box {
            padding: 1rem;
            border-radius: 10px;
            border: 1px solid #dddddd;
            margin-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🏝️ Mauritius Tourism Chatbot</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Your intelligent tourism assistant for exploring Mauritius.
        Ask about attractions, beaches, food, accommodation, transport,
        activities, culture, weather, itineraries and more.
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SESSION STATE
# =========================================================

# Local chatbot
if "bot" not in st.session_state:
    st.session_state.bot = TourismChatbot()


# Conversation history shown in Streamlit
if "history" not in st.session_state:
    st.session_state.history = []


# Gemini conversation memory
if "ai_thread_id" not in st.session_state:
    st.session_state.ai_thread_id = None


# Statistics
if "stats" not in st.session_state:
    st.session_state.stats = {
        "knowledge_base": 0,
        "ai_assistant": 0,
        "rate_limited": 0,
        "unanswered": 0,
        "errors": 0,
    }


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("🌴 Mauritius Tourism")

    st.write(
        "I can help you with a wide range of questions about "
        "travelling to and exploring Mauritius."
    )

    st.divider()

    st.subheader("💡 Try asking about")

    suggested_questions = [
        "What are the best beaches in Mauritius?",
        "What is the best time to visit Mauritius?",
        "What food should I try?",
        "How can I get around the island?",
        "What are the best activities for families?",
        "Can you plan a 7-day itinerary?",
        "What are some romantic places for couples?",
        "What can I do on a budget?",
    ]

    for question in suggested_questions:
        if st.button(
            question,
            key=f"suggestion_{question}",
            use_container_width=True,
        ):
            st.session_state.pending_question = question


    st.divider()

    st.subheader("📚 Built-in knowledge")

    topic_names = [
        name.replace("_", " ").title()
        for name in INTENTS
    ]

    st.caption(
        ", ".join(topic_names)
    )

    st.divider()

    st.subheader("📊 Chatbot statistics")

    stats = st.session_state.stats

    st.metric(
        "Knowledge-base answers",
        stats["knowledge_base"],
    )

    st.metric(
        "AI-generated answers",
        stats["ai_assistant"],
    )

    if stats["rate_limited"] > 0:
        st.metric(
            "Rate-limited requests",
            stats["rate_limited"],
        )

    if stats["unanswered"] > 0:
        st.metric(
            "Unanswered",
            stats["unanswered"],
        )

    st.divider()

    if st.button(
        "🗑️ Start New Conversation",
        use_container_width=True,
    ):

        # Clear conversation memory
        st.session_state.history = []

        # Reset Gemini conversation
        st.session_state.ai_thread_id = None

        # Clear pending question
        if "pending_question" in st.session_state:
            del st.session_state.pending_question

        st.rerun()


# =========================================================
# DISPLAY PREVIOUS CONVERSATION
# =========================================================

for item in st.session_state.history:

    role = item["role"]
    text = item["text"]
    source = item.get("source")

    with st.chat_message(role):

        st.markdown(text)

        if role == "assistant" and source:

            if source == "knowledge_base":

                st.caption(
                    "📚 Answered using the built-in tourism knowledge base."
                )

            elif source == "ai_assistant":

                st.caption(
                    "🤖 Answered by the AI tourism assistant."
                )

            elif source == "rate_limited":

                st.caption(
                    "⏳ AI service temporarily rate-limited."
                )

            elif source == "unanswered":

                st.caption(
                    "ℹ️ General fallback response."
                )

            elif source == "error":

                st.caption(
                    "⚠️ An error occurred while generating the response."
                )


# =========================================================
# GET USER QUESTION
# =========================================================

pending_question = st.session_state.pop(
    "pending_question",
    None,
)

typed_question = st.chat_input(
    "Ask me anything about Mauritius..."
)

prompt = typed_question or pending_question


# =========================================================
# PROCESS QUESTION
# =========================================================

if prompt:

    prompt = prompt.strip()

    if not prompt:
        st.warning("Please enter a question.")
        st.stop()


    # -----------------------------------------------------
    # Display user message immediately
    # -----------------------------------------------------

    with st.chat_message("user"):
        st.markdown(prompt)


    # -----------------------------------------------------
    # Add user question to history
    # -----------------------------------------------------

    st.session_state.history.append(
        {
            "role": "user",
            "text": prompt,
            "source": None,
        }
    )


    # -----------------------------------------------------
    # Generate response
    # -----------------------------------------------------

    reply = None
    source = "unanswered"


    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                # =================================================
                # LAYER 1 — LOCAL KNOWLEDGE BASE
                # =================================================

                debug_result = st.session_state.bot.respond_debug(
                    prompt
                )


                if debug_result["is_confident"]:

                    reply = debug_result["response"]
                    source = "knowledge_base"

                    st.session_state.stats[
                        "knowledge_base"
                    ] += 1


                # =================================================
                # LAYER 2 — GEMINI AI
                # =================================================

                elif llm_fallback.is_available():

                    llm_reply, new_thread_id = (
                        llm_fallback.answer(
                            prompt,
                            previous_interaction_id=(
                                st.session_state.ai_thread_id
                            ),
                        )
                    )


                    # -------------------------------------------------
                    # Rate limit
                    # -------------------------------------------------

                    if (
                        llm_reply
                        == llm_fallback.RATE_LIMIT_MESSAGE
                    ):

                        reply = llm_reply
                        source = "rate_limited"

                        st.session_state.stats[
                            "rate_limited"
                        ] += 1


                    # -------------------------------------------------
                    # Successful AI response
                    # -------------------------------------------------

                    elif llm_reply:

                        reply = llm_reply
                        source = "ai_assistant"

                        # Save conversation memory
                        if new_thread_id:

                            st.session_state.ai_thread_id = (
                                new_thread_id
                            )

                        st.session_state.stats[
                            "ai_assistant"
                        ] += 1


                    # -------------------------------------------------
                    # Empty AI response
                    # -------------------------------------------------

                    else:

                        reply = (
                            "I'm sorry, I couldn't generate an "
                            "answer right now. Please try "
                            "rephrasing your question."
                        )

                        source = "unanswered"

                        st.session_state.stats[
                            "unanswered"
                        ] += 1


                # =================================================
                # GEMINI NOT AVAILABLE
                # =================================================

                else:

                    reply = FALLBACK
                    source = "unanswered"

                    st.session_state.stats[
                        "unanswered"
                    ] += 1


            # -----------------------------------------------------
            # Unexpected application error
            # -----------------------------------------------------

            except Exception as exc:

                reply = (
                    "I'm sorry, something went wrong while "
                    "processing your question. Please try again."
                )

                source = "error"

                st.session_state.stats[
                    "errors"
                ] += 1

                print(
                    "[streamlit_app] Error:",
                    type(exc).__name__,
                    str(exc),
                )


        # =========================================================
        # DISPLAY RESPONSE
        # =========================================================

        if reply:
            st.markdown(reply)


        # =========================================================
        # SOURCE LABEL
        # =========================================================

        if source == "knowledge_base":

            st.caption(
                "📚 Answered using the built-in Mauritius "
                "tourism knowledge base."
            )

        elif source == "ai_assistant":

            st.caption(
                "🤖 Answered by the AI tourism assistant."
            )

        elif source == "rate_limited":

            st.caption(
                "⏳ The AI service is temporarily rate-limited. "
                "Please try again shortly."
            )

        elif source == "error":

            st.caption(
                "⚠️ The chatbot encountered an unexpected error."
            )

        elif source == "unanswered":

            st.caption(
                "ℹ️ General fallback response."
            )


    # -----------------------------------------------------
    # Save assistant response to history
    # -----------------------------------------------------

    st.session_state.history.append(
        {
            "role": "assistant",
            "text": reply,
            "source": source,
        }
    )
