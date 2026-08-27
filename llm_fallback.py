"""
Gemini AI fallback for the Mauritius Tourism Chatbot.

This module provides:
- Gemini API connection
- Availability checking
- AI-generated tourism responses
- Conversation memory using previous_interaction_id
- Error handling
"""

import streamlit as st
from google import genai


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_NAME = "gemini-3.7-flash"

RATE_LIMIT_MESSAGE = (
    "The AI assistant is temporarily unavailable because "
    "the service has reached its request limit. "
    "Please try again shortly."
)


SYSTEM_INSTRUCTION = """
You are an intelligent tourism assistant specialising in Mauritius.

Your main purpose is to help tourists and potential visitors
understand Mauritius and plan their trips.

You can answer questions about:

- Beaches
- Tourist attractions
- Hotels and accommodation
- Restaurants
- Mauritian food
- Activities and excursions
- Nature and wildlife
- Hiking
- Water sports
- Diving and snorkelling
- Culture
- History
- Local traditions
- Transportation
- Weather and seasons
- Family holidays
- Honeymoons
- Romantic travel
- Budget travel
- Luxury travel
- Itineraries
- Travel planning
- General tourism information about Mauritius

CONVERSATION MEMORY:

Remember relevant information from previous messages.

For example, if the user says:

"I am travelling with my family."

and later asks:

"What activities would you recommend?"

understand that the question refers to their family trip.

If the user says:

"What about those?"

use the previous conversation to determine what
"those" refers to.

If the user's question is genuinely unclear,
ask a short clarification question.

RESPONSE STYLE:

- Be friendly and professional.
- Focus on Mauritius tourism.
- Give practical recommendations.
- Use bullet points or numbered lists when helpful.
- Avoid unnecessarily long answers.
- Do not claim to have personally visited Mauritius.
- Do not invent specific prices, opening hours,
  schedules or other time-sensitive information.
- If information may have changed, tell the user
  that they should verify the latest information.

If the question is unrelated to Mauritius,
briefly explain that your main purpose is assisting
with Mauritius tourism.
"""


# =========================================================
# GEMINI CLIENT
# =========================================================

@st.cache_resource
def get_client():
    """
    Create and cache the Gemini client.
    """

    api_key = st.secrets.get("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured in "
            "Streamlit Secrets."
        )

    return genai.Client(api_key=api_key)


# =========================================================
# CHECK AVAILABILITY
# =========================================================

def is_available():
    """
    Check whether Gemini can be used.

    This function is intentionally lightweight.
    It checks whether the API key exists rather than
    making an unnecessary API request.
    """

    try:

        api_key = st.secrets.get("GEMINI_API_KEY")

        return bool(api_key)

    except Exception as exc:

        print(
            "[llm_fallback] Availability check failed:",
            type(exc).__name__,
            str(exc),
        )

        return False


# =========================================================
# GEMINI ANSWER
# =========================================================

def answer(
    question: str,
    previous_interaction_id=None,
):
    """
    Generate a Gemini response.

    Parameters
    ----------
    question : str
        User's current question.

    previous_interaction_id : str or None
        Previous Gemini interaction ID used to maintain
        conversation memory.

    Returns
    -------
    tuple
        (response_text, interaction_id)
    """

    if not question or not question.strip():

        return (
            "Please enter a question about Mauritius.",
            previous_interaction_id,
        )


    try:

        client = get_client()


        # =====================================================
        # FIRST CONVERSATION
        # =====================================================

        if previous_interaction_id is None:

            interaction = client.interactions.create(

                model=MODEL_NAME,

                input=question.strip(),

                system_instruction=SYSTEM_INSTRUCTION,
            )


        # =====================================================
        # CONTINUE CONVERSATION
        # =====================================================

        else:

            interaction = client.interactions.create(

                model=MODEL_NAME,

                input=question.strip(),

                previous_interaction_id=(
                    previous_interaction_id
                ),

                system_instruction=SYSTEM_INSTRUCTION,
            )


        # =====================================================
        # CHECK STATUS
        # =====================================================

        status = getattr(
            interaction,
            "status",
            None,
        )

        if status == "failed":

            print(
                "[llm_fallback] Gemini interaction failed."
            )

            return (
                None,
                previous_interaction_id,
            )


        # =====================================================
        # EXTRACT RESPONSE
        # =====================================================

        response_text = getattr(
            interaction,
            "output_text",
            None,
        )


        if response_text:

            response_text = response_text.strip()


        # =====================================================
        # FALLBACK RESPONSE EXTRACTION
        # =====================================================

        if not response_text:

            steps = getattr(
                interaction,
                "steps",
                [],
            ) or []


            text_parts = []


            for step in steps:

                step_type = getattr(
                    step,
                    "type",
                    None,
                )


                if step_type != "model_output":

                    continue


                content = getattr(
                    step,
                    "content",
                    [],
                ) or []


                for block in content:

                    block_type = getattr(
                        block,
                        "type",
                        None,
                    )


                    if block_type == "text":

                        text = getattr(
                            block,
                            "text",
                            None,
                        )


                        if text:

                            text_parts.append(
                                text
                            )


            response_text = "\n".join(
                text_parts
            ).strip()


        # =====================================================
        # EMPTY RESPONSE
        # =====================================================

        if not response_text:

            print(
                "[llm_fallback] Gemini returned "
                "an empty response."
            )

            return (
                None,
                previous_interaction_id,
            )


        # =====================================================
        # SUCCESS
        # =====================================================

        return (
            response_text,
            getattr(
                interaction,
                "id",
                previous_interaction_id,
            ),
        )


    # =========================================================
    # RATE LIMIT
    # =========================================================

    except Exception as exc:

        error_type = type(exc).__name__
        error_message = str(exc)

        print(
            "[llm_fallback] Gemini error:",
            error_type,
            error_message,
        )


        # Detect common rate-limit errors
        error_lower = error_message.lower()


        if (
            "429" in error_message
            or "rate limit" in error_lower
            or "quota" in error_lower
        ):

            return (
                RATE_LIMIT_MESSAGE,
                previous_interaction_id,
            )


        # Other errors
        return (
            None,
            previous_interaction_id,
        )
