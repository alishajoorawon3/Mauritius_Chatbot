"""
Gemini LLM layer for the Mauritius Tourism Chatbot.

This module provides:
1. Broad question answering beyond the TF-IDF knowledge base.
2. Conversation memory supplied by Streamlit.
3. Grounding using the chatbot's curated Mauritius knowledge base.
4. Safe handling of missing API keys and API errors.
"""

import os

from google import genai
from google.genai import types

from chatbot_core import INTENTS


MODEL = "gemini-3.7-flash"


SYSTEM_INSTRUCTION = """
You are the AI assistant for a Mauritius tourism chatbot.

Your role is to provide helpful, clear and practical information
for people planning to visit Mauritius.

IMPORTANT RULES:

1. Focus primarily on Mauritius tourism.
2. You may answer questions about:
   - attractions
   - beaches
   - hotels and accommodation
   - transport
   - food and restaurants
   - culture
   - history
   - geography
   - weather
   - activities
   - diving and snorkelling
   - hiking
   - honeymoon trips
   - family holidays
   - budgets
   - currency
   - visas
   - safety
   - internet/connectivity
   - shopping
   - nightlife
   - itineraries
   - travel planning
   - general information about Mauritius

3. You may answer reasonable general questions if they help
   the traveller understand Mauritius.

4. Use the conversation history to understand follow-up questions.

5. If the user says:
   "What about families?"
   "What about in the south?"
   "How much would that cost?"
   "Which one is better?"
   etc.,
   interpret the question in the context of the previous conversation.

6. Do not pretend that static information is current.
   For information that can change frequently, such as:
   - exchange rates
   - visa requirements
   - flight schedules
   - hotel prices
   - attraction opening hours
   - weather
   - current events
   clearly indicate that the user should verify the latest information
   from an official source.

7. Do not invent precise prices, opening hours, regulations or
   current statistics when you are uncertain.

8. When giving recommendations, explain briefly why the option
   may suit the traveller.

9. Keep answers reasonably concise and easy to read.

10. If the user gives travel preferences such as:
    - travelling with children
    - travelling as a couple
    - budget level
    - number of days
    - preferred region
    - interests
    remember those details during the current conversation and
    use them when answering later questions.

11. Never claim that you have booked a hotel, flight or activity.

12. For medical, legal or emergency matters, provide general
    information and recommend contacting the appropriate official
    or professional service.

The chatbot is a tourism information prototype developed for
academic research about Mauritius.
"""


def get_api_key():
    """Retrieve the Gemini API key from Streamlit secrets or environment."""

    try:
        import streamlit as st

        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]

    except Exception:
        pass

    return os.environ.get("GEMINI_API_KEY")


def is_available():
    """Return True when a Gemini API key is configured."""

    return bool(get_api_key())


def build_knowledge_context():
    """
    Convert the curated INTENTS knowledge base into a compact
    text context that can be supplied to Gemini.
    """

    sections = []

    for name, data in INTENTS.items():

        examples = data.get("examples", [])
        response = data.get("response", "")

        section = (
            f"TOPIC: {name}\n"
            f"EXAMPLE QUESTIONS: {', '.join(examples[:5])}\n"
            f"CURATED ANSWER: {response}\n"
        )

        sections.append(section)

    return "\n".join(sections)


KNOWLEDGE_CONTEXT = build_knowledge_context()


def _convert_history(history):
    """
    Convert Streamlit conversation history into Gemini Content objects.

    Expected history format:

    [
        {"role": "user", "content": "..."},
        {"role": "model", "content": "..."},
        ...
    ]
    """

    converted = []

    for item in history:

        role = item.get("role")

        if role not in {"user", "model"}:
            continue

        content = item.get("content", "")

        if not content:
            continue

        converted.append(
            types.Content(
                role=role,
                parts=[
                    types.Part(text=content)
                ],
            )
        )

    return converted


def answer(query, history=None):
    """
    Generate a Gemini response using the supplied conversation history.

    Returns:

        (response_text, True)

    on success.

        (None, False)

    on failure.
    """

    api_key = get_api_key()

    if not api_key:
        return None, False

    if history is None:
        history = []

    try:

        client = genai.Client(api_key=api_key)

        # Keep the context manageable.
        # 12 messages = approximately the latest 6 exchanges.
        recent_history = history[-12:]

        gemini_history = _convert_history(recent_history)

        chat = client.chats.create(
            model=MODEL,
            history=gemini_history,
        )

        contextual_prompt = f"""
CURATED MAURITIUS KNOWLEDGE BASE:

{KNOWLEDGE_CONTEXT}

---

CURRENT USER QUESTION:

{query}

Answer the user's question using the conversation history,
the curated knowledge base above, and your general knowledge.

If the question refers to something mentioned earlier,
continue the conversation naturally.
"""

        response = chat.send_message(
            contextual_prompt
        )

        text = (response.text or "").strip()

        if not text:
            return None, False

        return text, True

    except Exception as exc:

        print(
            f"[llm_fallback] Gemini error: "
            f"{type(exc).__name__}: {exc}"
        )

        return None, False
