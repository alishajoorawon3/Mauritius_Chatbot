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
You can answer questions about: beaches, tourist attractions, hotels and
accommodation, restaurants, Mauritian food, activities and excursions,
nature and wildlife, hiking, water sports, diving and snorkelling, culture,
history, local traditions, transportation, weather and seasons, family
holidays, honeymoons, romantic travel, budget travel, luxury travel,
itineraries, travel planning, and general tourism information.
CONVERSATION MEMORY:
Remember relevant information from previous messages. For example, if the
user says "I am travelling with my family." and later asks "What activities
would you recommend?", understand that the question refers to their family
trip. If the user says "What about those?", use the previous conversation to
determine what "those" refers to. If the user's question is genuinely
unclear, ask a short clarification question.
RESPONSE STYLE:
- Be friendly and professional.
- Focus on Mauritius tourism.
- Give practical recommendations.
- Use bullet points or numbered lists when helpful.
- Avoid unnecessarily long answers.
- Do not claim to have personally visited Mauritius.
- Do not invent specific prices, opening hours, schedules or other
  time-sensitive information.
If the question is unrelated to Mauritius, briefly explain that your main
purpose is assisting with Mauritius tourism.
"""
# =========================================================
# GEMINI CLIENT
# =========================================================
@st.cache_resource
def get_client():
    """Create and cache the Gemini client."""
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured in Streamlit Secrets."
        )
    return genai.Client(api_key=api_key)
# =========================================================
# CHECK AVAILABILITY
# =========================================================
def is_available():
    """Intentionally lightweight: checks the key exists rather than
    making an unnecessary API request."""
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
        return bool(api_key)
    except Exception as exc:
        print("[llm_fallback] Availability check failed:",
              type(exc).__name__, str(exc))
        return False
# =========================================================
# GEMINI ANSWER
# =========================================================
def answer(question: str, previous_interaction_id=None):
    """Returns (response_text, interaction_id)."""
    if not question or not question.strip():
        return ("Please enter a question about Mauritius.",
                previous_interaction_id)
    try:
        client = get_client()
        if previous_interaction_id is None:
            interaction = client.interactions.create(
                model=MODEL_NAME,
                input=question.strip(),
                system_instruction=SYSTEM_INSTRUCTION,
            )
        else:
            interaction = client.interactions.create(
                model=MODEL_NAME,
                input=question.strip(),
                previous_interaction_id=previous_interaction_id,
                system_instruction=SYSTEM_INSTRUCTION,
            )
        status = getattr(interaction, "status", None)
        if status == "failed":
            print("[llm_fallback] Gemini interaction failed.")
            return (None, previous_interaction_id)
        response_text = getattr(interaction, "output_text", None)
        if response_text:
            response_text = response_text.strip()
        # Fallback extraction: walk the steps for model_output text blocks.
        if not response_text:
            steps = getattr(interaction, "steps", []) or []
            text_parts = []
            for step in steps:
                if getattr(step, "type", None) != "model_output":
                    continue
                for block in getattr(step, "content", []) or []:
                    if getattr(block, "type", None) == "text":
                        text = getattr(block, "text", None)
                        if text:
                            text_parts.append(text)
            response_text = "\n".join(text_parts).strip()
        if not response_text:
            return (None, previous_interaction_id)
        return (response_text,
                getattr(interaction, "id", previous_interaction_id))
    except Exception as exc:
        print("[llm_fallback] ERROR:", type(exc).__name__, str(exc))
        return (None, previous_interaction_id)
