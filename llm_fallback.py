"""
Gemini fallback for the Mauritius Tourism Chatbot.

This module:
- Uses Gemini for questions not confidently answered by the local knowledge base.
- Supports multi-turn conversation memory using previous_interaction_id.
- Uses Streamlit secrets for the Gemini API key.
- Returns useful error information instead of silently failing.
"""

import streamlit as st
from google import genai


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_NAME = "gemini-3.7-flash"

SYSTEM_INSTRUCTION = """
You are the intelligent tourism assistant for Mauritius.

Your purpose is to help tourists and potential visitors understand
Mauritius and plan trips.

You can answer questions about:
- Tourist attractions
- Beaches and coastal areas
- Hotels and accommodation
- Restaurants and Mauritian food
- Activities and excursions
- Nature and wildlife
- Hiking and outdoor activities
- Culture and history
- Local traditions
- Transportation
- Weather and seasons
- Tourist safety
- Family travel
- Honeymoons and romantic travel
- Budget travel
- Luxury travel
- Suggested itineraries
- Travel planning
- General information about Mauritius

Conversation behaviour:
- Remember relevant information from previous messages.
- If the user asks a follow-up question such as "What about families?",
  "How much would that cost?", or "What about the south?",
  use the previous conversation to understand what they mean.
- Ask a short clarification question when the user's request is genuinely
  ambiguous.
- Give practical and useful answers rather than extremely generic answers.
- When appropriate, organise information using bullet points or numbered lists.
- Keep answers concise enough for a chatbot but provide enough detail to be useful.
- Do not invent specific prices, opening hours, transport schedules or
  other time-sensitive facts.
- If information may have changed, clearly indicate that the user should
  verify the latest information.
- Do not claim to have personally visited places.
- Be friendly, professional and tourism-focused.

Important:
The chatbot is designed specifically around Mauritius tourism.
When the question is unrelated to Mauritius, answer briefly if it is useful,
but explain that your main purpose is assisting with Mauritius tourism.
"""


# ---------------------------------------------------------
# Create Gemini client
# ---------------------------------------------------------

@st.cache_resource
def get_gemini_client():
    """
    Create and cache the Gemini client.

    The API key is stored in Streamlit Cloud Secrets and is never
    placed directly in the GitHub repository.
    """

    try:
        api_key = st.secrets["GEMINI_API_KEY"]

        if not api_key:
            raise ValueError("GEMINI_API_KEY is empty.")

        return genai.Client(api_key=api_key)

    except KeyError:
        raise RuntimeError(
            "GEMINI_API_KEY was not found in Streamlit Secrets."
        )

    except Exception as exc:
        raise RuntimeError(
            f"Could not initialise Gemini client: {exc}"
        )


# ---------------------------------------------------------
# Gemini response function
# ---------------------------------------------------------

def answer(question: str, previous_interaction_id=None):
    """
    Send a question to Gemini.

    Returns:
        (answer_text, interaction_id)

    The interaction ID is stored by Streamlit so that future
    questions can continue the same conversation.
    """

    if not question or not question.strip():
        return (
            "Please enter a question about Mauritius.",
            previous_interaction_id,
        )

    try:
        client = get_gemini_client()

        # First interaction
        if previous_interaction_id is None:

            interaction = client.interactions.create(
                model=MODEL_NAME,
                input=question.strip(),
                system_instruction=SYSTEM_INSTRUCTION,
            )

        # Continue existing conversation
        else:

            interaction = client.interactions.create(
                model=MODEL_NAME,
                input=question.strip(),
                previous_interaction_id=previous_interaction_id,
                system_instruction=SYSTEM_INSTRUCTION,
            )

        # -------------------------------------------------
        # Check interaction status
        # -------------------------------------------------

        status = getattr(interaction, "status", None)

        if status == "failed":
            return (
                "I was unable to generate an answer at the moment. "
                "Please try your question again.",
                previous_interaction_id,
            )

        # -------------------------------------------------
        # Extract response
        # -------------------------------------------------

        response_text = getattr(interaction, "output_text", None)

        if response_text:
            response_text = response_text.strip()

        # Safety fallback in case output_text is unavailable
        if not response_text:

            steps = getattr(interaction, "steps", []) or []

            text_parts = []

            for step in steps:

                step_type = getattr(step, "type", None)

                if step_type != "model_output":
                    continue

                content = getattr(step, "content", []) or []

                for block in content:

                    block_type = getattr(block, "type", None)

                    if block_type == "text":

                        text = getattr(block, "text", None)

                        if text:
                            text_parts.append(text)

            response_text = "\n".join(text_parts).strip()

        # -------------------------------------------------
        # Final validation
        # -------------------------------------------------

        if not response_text:

            return (
                "I couldn't generate a response. "
                "Please try asking the question in another way.",
                previous_interaction_id,
            )

        # Return answer + ID for conversation memory
        return response_text, interaction.id

    except Exception as exc:

        # IMPORTANT:
        # Do not silently hide the error.
        # Returning the error allows streamlit_app.py to display
        # a useful diagnostic message.

        error_message = (
            f"Gemini error: {type(exc).__name__}: {exc}"
        )

        print(f"[llm_fallback] {error_message}")

        return (
            error_message,
            previous_interaction_id,
        )
