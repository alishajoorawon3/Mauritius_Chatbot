"""
LLM fallback layer for the Mauritius Chatbot - backed by Google's Gemini
API (free tier: a Google account is enough, no credit card needed for
casual/demo-level use).

When the offline TF-IDF matcher (chatbot_core.py) isn't confident it knows
the topic, this module asks Gemini to answer instead, grounded in a curated
document of Mauritius facts (built from the same INTENTS knowledge base) so
it stays consistent with the offline layer.

Multi-turn memory: Gemini's Interactions API supports chaining follow-up
questions onto a previous exchange via `previous_interaction_id`, so a
question like "what about for families?" can be understood as a follow-up
to whatever was just discussed, instead of being answered from scratch with
no context. The caller (streamlit_app.py) is responsible for remembering
the most recent interaction id (in st.session_state) and passing it back in
on the next call - this module doesn't persist anything itself. Note this
only threads together turns that actually reach the AI layer; a question
answered by the offline matcher in between doesn't get added to the AI's
memory, since it never talks to Gemini at all - a known limitation of the
hybrid architecture, worth a mention in a limitations section.

Requires a Gemini API key, supplied as a Streamlit secret or environment
variable named GEMINI_API_KEY. If no key is configured, or the API call
fails for any reason, `answer()` returns (None, None) so the caller can
fall back to the static offline fallback message instead of crashing.
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


def _call_gemini(query, api_key, previous_interaction_id):
    from google import genai
    client = genai.Client(api_key=api_key)
    interaction = client.interactions.create(
        model=MODEL,
        input=query,
        system_instruction=SYSTEM_INSTRUCTION,
        previous_interaction_id=previous_interaction_id,
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
    interaction_id = getattr(interaction, "id", None)

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

    return (text, interaction_id) if text else (None, None)


RATE_LIMIT_MESSAGE = (
    "The AI layer is temporarily rate-limited by Google's free-tier quota "
    "(too many questions in a short window) - please wait a minute and try "
    "again. The knowledge-base topics in the sidebar still answer instantly "
    "in the meantime."
)


def _is_rate_limit_error(exc):
    """Google's free tier caps requests per minute/day; bursts of testing
    (or, for a real deployed app, several visitors asking questions close
    together) can trip this. It surfaces as a 429/RESOURCE_EXHAUSTED error
    from the API - worth telling the user plainly rather than showing the
    generic 'I'm not sure about that' message, which reads as if the
    knowledge base just doesn't cover the topic when actually it's a
    temporary capacity issue that will resolve itself shortly."""
    msg = str(exc).lower()
    return "429" in str(exc) or "resource_exhausted" in msg or "too many requests" in msg or "rate limit" in msg


def answer(query, previous_interaction_id=None):
    """Returns (text, interaction_id). `text` is None if the LLM fallback
    isn't configured or the call fails outright - the caller should show
    the static offline fallback message in that case. If the failure is
    specifically a rate limit, `text` is RATE_LIMIT_MESSAGE instead (a real,
    honest string to show the user, not a signal to fall back further) and
    `interaction_id` is None. On success, `interaction_id` is the id of
    this exchange on Gemini's side; pass it back in as
    `previous_interaction_id` on the *next* call (from the same
    conversation) so Gemini remembers what was just discussed and can
    handle follow-up questions naturally. Pass None for a fresh
    conversation with no prior context."""
    api_key = _get_api_key()
    if not api_key:
        return (None, None)

    try:
        return _call_gemini(query, api_key, previous_interaction_id)
    except Exception as exc:
        if _is_rate_limit_error(exc):
            print(f"[llm_fallback] Rate limited: {type(exc).__name__}: {exc}")
            return (RATE_LIMIT_MESSAGE, None)

        # If we were continuing a thread and it failed for some other
        # reason, the thread id itself might be stale/invalid (expired, or
        # from a session that no longer exists on Gemini's side) - retry
        # once as a fresh conversation rather than let one bad id
        # permanently break every future turn. (Retrying a rate limit
        # immediately would just hit the same 429 again, hence the check
        # above happens first and returns early instead of retrying here.)
        if previous_interaction_id is not None:
            print(f"[llm_fallback] Gemini call failed with previous_interaction_id="
                  f"{previous_interaction_id!r} ({type(exc).__name__}: {exc}); "
                  f"retrying as a fresh conversation.")
            try:
                return _call_gemini(query, api_key, None)
            except Exception as exc2:
                if _is_rate_limit_error(exc2):
                    print(f"[llm_fallback] Rate limited on retry: {type(exc2).__name__}: {exc2}")
                    return (RATE_LIMIT_MESSAGE, None)
                print(f"[llm_fallback] Retry without thread also failed: "
                      f"{type(exc2).__name__}: {exc2}")
                return (None, None)

        # Any other failure (bad key, network issue, unexpected response
        # shape) degrades gracefully to the offline fallback instead of
        # crashing the app. Logged (not raised) so the real cause is
        # visible in Streamlit Cloud's log viewer (Manage app -> logs)
        # without breaking the user-facing experience.
        print(f"[llm_fallback] Gemini call failed: {type(exc).__name__}: {exc}")
        return (None, None)
