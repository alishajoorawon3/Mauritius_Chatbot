"""
LLM fallback layer for the Mauritius Tourism Chatbot - now backed by
Google's Gemini API (free tier: a Google account is enough, no credit
card needed for casual/demo-level use).

When the offline TF-IDF matcher (chatbot_core.py) isn't confident it knows
the topic, this module asks Gemini to answer instead, grounded in a curated
document of Mauritius travel facts (built from the same INTENTS knowledge
base, plus a few general facts) so it stays on-topic and doesn't invent
specifics it can't know (e.g. real-time prices or forecasts).

Requires a Gemini API key, supplied as a Streamlit secret or environment
variable named GEMINI_API_KEY. If no key is configured, or the API call
fails for any reason, `answer()` returns None so the caller can fall back
to the static offline fallback message instead of crashing.
"""

import os
from chatbot_core import INTENTS

MODEL = "gemini-3.7-flash"
# Gemini 3.7 Flash spends part of its generation budget on internal
# "thinking" before writing the visible answer. Give it plenty of room
# (1024) and keep thinking_level low, since this task - a short, grounded
# factual answer - doesn't need deep reasoning; without both of these the
# visible reply can get cut off mid-sentence once the budget runs out.
MAX_OUTPUT_TOKENS = 1024
THINKING_LEVEL = "low"

GENERAL_FACTS = """
Country overview: Mauritius is an island nation in the Indian Ocean, east of
Madagascar. Its capital is Port Louis. It has a multicultural population
(Indo-Mauritian, Creole, Sino-Mauritian, Franco-Mauritian communities) and a
mix of Hindu, Christian, Muslim, and Buddhist traditions. Traffic drives on
the left. The timezone is GMT+4 (no daylight saving).
""".strip()


def _build_facts_document():
    """Concatenate every intent's canonical answer into one grounding
    document, so the LLM has the same underlying facts as the offline
    matcher and the two layers don't contradict each other."""
    sections = [GENERAL_FACTS]
    for name, data in INTENTS.items():
        title = name.replace("_", " ").title()
        sections.append(f"{title}: {data['response']}")
    return "\n\n".join(sections)


FACTS_DOCUMENT = _build_facts_document()

SYSTEM_INSTRUCTION = f"""You are a helpful, friendly assistant for tourists planning a trip to \
Mauritius, embedded in a small demo chatbot.

Ground your answers in the facts below. You may also use well-established, \
uncontroversial general knowledge about Mauritius to fill small gaps, but do \
NOT invent specific numbers, prices, business names, or live details (exact \
exchange rates, real-time weather, today's opening hours, a specific \
restaurant table) that you cannot actually know - say plainly that you don't \
have that live information and suggest how the traveller could find it \
(official government site, their airline, their hotel, etc.) instead.

If the question has nothing to do with travelling to Mauritius, say briefly \
that you can only help with Mauritius travel questions.

Keep answers concise and conversational: 2-4 sentences unless the question \
genuinely needs a short list.

=== MAURITIUS TRAVEL FACTS ===
{FACTS_DOCUMENT}
=== END OF FACTS ==="""


def _get_api_key():
    try:
        import streamlit as st
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY")


def is_available():
    return bool(_get_api_key())


def answer(query, history=None):
    """Returns Gemini's answer to `query`, or None if the LLM fallback isn't
    configured or the call fails. `history` is accepted for interface
    compatibility but this lightweight demo only sends the current
    question (see SETUP.md for extending this to multi-turn context via
    previous_interaction_id)."""
    api_key = _get_api_key()
    if not api_key:
        return None

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        interaction = client.interactions.create(
            model=MODEL,
            input=query,
            system_instruction=SYSTEM_INSTRUCTION,
            generation_config={
                "max_output_tokens": MAX_OUTPUT_TOKENS,
                "temperature": 0.4,
                "thinking_level": THINKING_LEVEL,
            },
        )
        text = getattr(interaction, "output_text", None)
        if not text and getattr(interaction, "outputs", None):
            text = interaction.outputs[-1].text
        return text.strip() if text else None
    except Exception as exc:
        # Any failure (bad key, network issue, rate limit, unexpected
        # response shape) degrades gracefully to the offline fallback
        # instead of crashing the app. Logged (not raised) so the real
        # cause is visible in Streamlit Cloud's log viewer (Manage app ->
        # logs) without breaking the user-facing experience.
        print(f"[llm_fallback] Gemini call failed: {type(exc).__name__}: {exc}")
        return None
