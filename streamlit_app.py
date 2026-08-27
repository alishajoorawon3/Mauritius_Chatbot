import streamlit as st

from chatbot_core import TourismChatbot, INTENTS, FALLBACK
import llm_fallback


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Mauritius Tourism Chatbot",
    page_icon="🏝️",
    layout="centered",
)


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("🏝️ Mauritius Tourism Chatbot")

st.caption(
    "Ask me about visiting Mauritius — attractions, beaches, "
    "hotels, transport, food, activities, weather, culture, "
    "budgets and more."
)


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

if "bot" not in st.session_state:
    st.session_state.bot = TourismChatbot()


if "messages" not in st.session_state:
    st.session_state.messages = []


if "stats" not in st.session_state:
    st.session_state.stats = {
        "knowledge_base": 0,
        "ai_assistant": 0,
        "fallback": 0,
    }


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def is_follow_up_question(question):
    """
    Detect whether a question is likely referring to
    something discussed previously.
    """

    follow_up_words = {
        "it",
        "that",
        "those",
        "them",
        "they",
        "this",
        "these",
        "which one",
        "what about",
        "how about",
        "and what",
        "what if",
        "how much",
        "how many",
        "which is better",
        "what would you recommend",
        "would that",
        "is that",
        "and for",
        "for families",
        "for couples",
        "for children",
    }

    question_lower = question.lower()

    return any(
        phrase in question_lower
        for phrase in follow_up_words
    )


def is_dynamic_question(question):
    """
    Identify questions involving information that can change.
    """

    dynamic_terms = {
        "current",
        "today",
        "now",
        "latest",
        "recent",
        "this week",
        "this month",
        "this year",
        "exchange rate",
        "price",
        "prices",
        "cost today",
        "flight",
        "flights",
        "opening hours",
        "open today",
        "closed today",
        "weather today",
        "weather tomorrow",
        "cyclone now",
        "visa requirement",
        "visa requirements",
    }

    question_lower = question.lower()

    return any(
        term in question_lower
        for term in dynamic_terms
    )


def should_use_ai(question, debug_result):
    """
    Decide whether Gemini should handle the question.

    Gemini is preferred when:
    1. The question is a follow-up.
    2. The question asks for dynamic/current information.
    3. The TF-IDF classifier is uncertain.
    """

    if len(st.session_state.messages) > 0:

        if is_follow_up_question(question):
            return True

    if is_dynamic_question(question):
        return True

    if not debug_result["is_confident"]:
        return True

    return False


def get_conversation_history():
    """
    Return the stored conversation in Gemini-compatible format.

    Only actual user/assistant messages are included.
    """

    history = []

    for message in st.session_state.messages:

        role = message["role"]

        if role == "user":
            history.append(
                {
                    "role": "user",
                    "content": message["content"],
                }
            )

        elif role == "assistant":
            history.append(
                {
                    "role": "model",
                    "content": message["content"],
                }
            )

    return history


# ---------------------------------------------------------
# DISPLAY PREVIOUS CONVERSATION
# ---------------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant":

            source = message.get("source")

            if source == "knowledge_base":

                st.caption(
                    "📚 Answered using the Mauritius knowledge base"
                )

            elif source == "ai_assistant":

                st.caption(
                    "🤖 Answered using the AI assistant"
                )


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.subheader("💡 Topics")

    topic_names = [
        name.replace("_", " ").title()
        for name in INTENTS
    ]

    for topic in topic_names:

        st.write(f"• {topic}")

    st.divider()

    st.subheader("🧠 Conversation")

    st.write(
        "The chatbot remembers relevant information "
        "from the current conversation so that follow-up "
        "questions can be understood in context."
    )

    if st.button(
        "🗑️ Clear conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.session_state.stats = {
            "knowledge_base": 0,
            "ai_assistant": 0,
            "fallback": 0,
        }

        st.rerun()


# ---------------------------------------------------------
# CHAT INPUT
# ---------------------------------------------------------

prompt = st.chat_input(
    "Ask me about visiting Mauritius..."
)


if prompt:

    # -----------------------------------------------------
    # SAVE USER MESSAGE
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):

        st.markdown(prompt)


    # -----------------------------------------------------
    # CLASSIFY QUESTION
    # -----------------------------------------------------

    debug_result = (
        st.session_state.bot.respond_debug(prompt)
    )


    # -----------------------------------------------------
    # DECIDE WHETHER TO USE GEMINI
    # -----------------------------------------------------

    use_ai = should_use_ai(
        prompt,
        debug_result,
    )


    reply = None
    source = None


    # -----------------------------------------------------
    # GEMINI
    # -----------------------------------------------------

    if use_ai and llm_fallback.is_available():

        conversation_history = (
            get_conversation_history()
        )

        # Remove the current user question because
        # it will be sent separately to Gemini.
        if conversation_history:
            conversation_history = conversation_history[:-1]


        llm_reply, success = llm_fallback.answer(
            prompt,
            history=conversation_history,
        )


        if success and llm_reply:

            reply = llm_reply

            source = "ai_assistant"

            st.session_state.stats[
                "ai_assistant"
            ] += 1


    # -----------------------------------------------------
    # KNOWLEDGE BASE
    # -----------------------------------------------------

    if reply is None and debug_result["is_confident"]:

        reply = debug_result["response"]

        source = "knowledge_base"

        st.session_state.stats[
            "knowledge_base"
        ] += 1


    # -----------------------------------------------------
    # FINAL FALLBACK
    # -----------------------------------------------------

    if reply is None:

        reply = (
            "I'm not completely sure about that. "
            "I can help with Mauritius travel planning, "
            "including accommodation, attractions, "
            "transport, food, activities, weather, "
            "visa information and more."
        )

        source = "fallback"

        st.session_state.stats[
            "fallback"
        ] += 1


    # -----------------------------------------------------
    # SAVE ASSISTANT RESPONSE
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": reply,
            "source": source,
        }
    )


    # -----------------------------------------------------
    # DISPLAY RESPONSE
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        st.markdown(reply)

        if source == "knowledge_base":

            st.caption(
                "📚 Answered using the Mauritius knowledge base"
            )

        elif source == "ai_assistant":

            st.caption(
                "🤖 Answered using the AI assistant"
            )
