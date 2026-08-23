"""
LLM fallback layer for the Mauritius Tourism Chatbot.

When the offline TF-IDF matcher (chatbot_core.py) isn't confident it knows
the topic, this module asks Claude to answer instead, grounded in a curated
document of Mauritius travel facts (built from the same INTENTS knowledge
base, plus a few general facts) so it stays on-topic and doesn't invent
specifics it can't know (e.g. real-time prices or forecasts).

Requires an Anthropic API key, supplied as a Streamlit secret or
environment variable named ANTHROPIC_API_KEY. If no key is configured, or
the API call fails for any reason, `answer()` returns None so the caller
can fall back to the static offline fallback message instead of crashing.
"""

import os
from chatbot_core import INTENTS

MODEL = "claude-3-5-haiku-latest"
MAX_TOKENS = 300

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

SYSTEM_PROMPT = f"""You are a helpful, friendly assistant for tourists planning a trip to \
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
        if "ANTHROPIC_API_KEY" in st.secrets:
            return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        pass
    return os.environ.get("ANTHROPIC_API_KEY")


def is_available():
    return bool(_get_api_key())


def answer(query, history=None):
    """Returns Claude's answer to `query`, or None if the LLM fallback isn't
    configured or the call fails. `history` is an optional list of
    {"role": "user"|"assistant", "content": str} for basic multi-turn
    context; kept short since this is a lightweight demo."""
    api_key = _get_api_key()
    if not api_key:
        return None

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        messages = list(history or [])
        messages.append({"role": "user", "content": query})
        resp = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        return resp.content[0].text.strip()
    except Exception:
        # Any failure (bad key, network issue, rate limit, unexpected
        # response shape) degrades gracefully to the offline fallback
        # instead of crashing the app.
        return None
