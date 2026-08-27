"""
Intent-based tourism chatbot prototype - v2.
Fixes found by the functional evaluation (Table 4.4) of the v1 prototype:
  1. "mauritius"/"island" dominated almost every match -> added as domain stopwords.
  2. Vocabulary was only 39 words across 8 topics -> expanded to 34 topics with
     8-10 natural example phrasings each.
  3. A single typo (e.g. "atractions") produced either a wrong confident match or
     an accidental fallback -> added fuzzy correction against the fitted vocabulary.
  4. A lone top score was trusted even when meaningless -> now aggregates by
     intent (max over that intent's examples) and requires both a minimum score
     AND a minimum margin over the runner-up intent before answering.
  5. Anything still not confident enough no longer just says "please rephrase" -
     the caller (streamlit_app.py) can hand it to an LLM fallback instead.
  6. A dedicated "recent_cyclones" topic was added after a real question ("when
     was the last cyclone in Mauritius?") was wrongly matched to the general
     "weather" topic.
"""
import re
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text as sk_text
from sklearn.metrics.pairwise import cosine_similarity
_TOKEN_RE = re.compile(r"[a-zA-Z]{2,}")
def simple_stem(word):
    """Tiny dependency-free stemmer (no nltk download needed) so that
    'attractions'/'attraction', 'hotels'/'hotel', 'activities'/'activity'
    etc. collapse to the same vocabulary entry."""
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("es") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    if word.endswith("ing") and len(word) > 6:
        return word[:-3]
    return word
def _tokenize(text_):
    return _TOKEN_RE.findall(text_.lower())
# ---------------------------------------------------------------- intents
# Each intent: a handful of natural example phrasings + one canonical answer.
INTENTS = {
    "best_time_to_visit": {
        "examples": [
            "when is the best time to visit mauritius",
            "best season to travel to mauritius",
            "when should i go to mauritius",
            "which months are best for a holiday here",
            "what time of year has the nicest weather for a visit",
            "when would you recommend visiting",
            "is december a good month to visit",
            "what's the ideal time of year to travel here",
        ],
        "response": ("Mauritius is warm year-round. The driest, sunniest months "
                     "are May to December; January to March is hotter and more "
                     "humid, and is the main cyclone season."),
    },
    "attractions": {
        "examples": [
            "what are the best attractions in mauritius",
            "things to do in mauritius",
            "tourist spots in mauritius",
            "what sights shouldn't i miss",
            "top places to visit here",
            "what are must-see locations on the island",
            "recommend some must-visit spots",
            "where can i go sightseeing",
        ],
        "response": ("Popular attractions include Le Morne Brabant, Chamarel Seven "
                     "Coloured Earths, Black River Gorges National Park, Ile aux "
                     "Cerfs, and the capital Port Louis with its waterfront and "
                     "Central Market."),
    },
    "transport": { ... },
    "accommodation": { ... },
    "currency": { ... },
    "weather": { ... },
    "recent_cyclones": { ... },
    "food": { ... },
    "language": { ... },
    # ... 26 further topics follow exactly the same shape.
}
