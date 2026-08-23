"""
LLM fallback layer for the Mauritius Chatbot - backed by Google's Gemini
API (free tier: a Google account is enough, no credit card needed for
casual/demo-level use).

When the offline TF-IDF matcher (chatbot_core.py) isn't confident it knows
the topic, this module asks Gemini to answer instead. It's grounded in a
curated document of Mauritius facts (built from the same INTENTS knowledge
base) so it stays consistent with the offline layer, but it's also
instructed to draw on its own general knowledge so it can handle genuinely
open-ended questions (history, culture, current affairs, "when was the last
cyclone", etc.), not just the pre-written travel topics.

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
# and keep thinking_level low, since this task - a short, grounded factual
# answer - doesn't need deep reasoning; without both of these the visible
# reply can get cut off mid-sentence (or come back empty) once the budget
# runs out. Raised from 2048 after the broader "answer almost anything"
# system instruction made some genuinely open-ended questions come back
# with empty text - the extra instructions plus a longer, more effortful
# answer apparently pushed a few responses right up against the old limit.
MAX_OUTPUT_TOKENS = 3072
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

SYSTEM_INSTRUCTION = f"""You are a knowledgeable, friendly assistant for anyone curious \
about Mauritius - tourists planning a trip, but also people asking general \
questions about the country - embedded in a small demo chatbot.

Answer using the curated facts below AND your own general knowledge about \
Mauritius: history, geography, culture, wildlife, politics, sport, current \
affairs, or anything else someone might reasonably ask. Don't limit yourself \
to the facts list below - it exists to keep you consistent with the rest of \
the app, not to cap what you're willing to talk about. Prefer being helpful \
and specific over being overly cautious.

The one thing to be careful about is live or rapidly-changing information \
you cannot actually verify right now - today's exact weather, current \
exchange rates, this week's cyclone status, live prices, opening hours, or a \
specific business's current availability. For those, give your best general \
understanding (typical patterns, recent history, etc.) but say plainly that \
it may be out of date, and point to an authoritative source (the official \
Mauritian government site, Mauritius Meteorological Services, the \
traveller's airline or hotel) for the current figure.

If the question has nothing at all to do with Mauritius, say briefly that \
you're focused on Mauritius-related questions.

Keep answers concise and conversational: 2-4 sentences, unless the question \
genuinely needs a short list or a bit more detail to actually answer it.

=== MAURITIUS REFERENCE FACTS ===
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
        outputs = getattr(interaction, "outputs", None) or []
        # Concatenate every text-bearing output block rather than just the
        # last one - if the model returns thinking/text as separate blocks,
        # taking only outputs[-1] could silently drop earlier text content.
        joined = "".join(getattr(o, "text", "") or "" for o in outputs)
        text = joined.strip() or (getattr(interaction, "output_text", None) or "").strip()

        if not text:
            # A successful call with empty visible text (no exception raised)
            # is a distinct failure mode from a crash - usually the token
            # budget ran out during "thinking" before any visible text was
            # written, or the response was blocked/filtered. Log the details
            # so this is visible in Streamlit Cloud's log viewer (Manage
            # app -> logs) instead of silently degrading to the generic
            # offline fallback message with no clue why.
            finish_reason = getattr(interaction, "finish_reason", None) or getattr(interaction, "stop_reason", None)
            usage = getattr(interaction, "usage", None)
            print(f"[llm_fallback] Empty response for query={query!r}: "
                  f"len(outputs)={len(outputs)} finish_reason={finish_reason!r} "
                  f"usage={usage!r} "
                  f"output_types={[type(o).__name__ for o in outputs]!r}")

        return text if text else None
    except Exception as exc:
        # Any failure (bad key, network issue, rate limit, unexpected
        # response shape) degrades gracefully to the offline fallback
        # instead of crashing the app. Logged (not raised) so the real
        # cause is visible in Streamlit Cloud's log viewer (Manage app ->
        # logs) without breaking the user-facing experience.
        print(f"[llm_fallback] Gemini call failed: {type(exc).__name__}: {exc}")
        return None
