"""
Intent-based tourism chatbot prototype - v2 (Streamlit Cloud fix).

This version guarantees the imports used by streamlit_app.py:
    from chatbot_core import TourismChatbot, INTENTS, FALLBACK

Design (unchanged from the evaluation):
  1. "mauritius"/"island" are domain stopwords so they don't dominate matches.
  2. 30+ tourism topics, each with several natural example phrasings.
  3. Fuzzy typo correction against the fitted vocabulary (difflib).
  4. Scores are aggregated by intent (max over that intent's examples).
  5. An answer is only trusted if BOTH a minimum score AND a minimum margin
     over the runner-up intent are met; otherwise the caller (streamlit_app.py)
     can hand the question to the Gemini AI fallback.
"""
import re
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text as sk_text
from sklearn.metrics.pairwise import cosine_similarity

_TOKEN_RE = re.compile(r"[a-zA-Z]{2,}")

MIN_SCORE = 0.30      # absolute minimum similarity to trust a match
MIN_MARGIN = 0.06     # required gap over the runner-up intent
DOMAIN_STOPWORDS = frozenset(["mauritius", "island", "mauritian"])
# Words that are in sklearn's stopword list but that we must keep.
PROTECTED = frozenset(["do", "can", "go", "best", "food", "cyclone"])


def simple_stem(word):
    """Tiny dependency-free stemmer so 'attraction'/'attractions', 'hotel'/
    'hotels', 'activity'/'activities' collapse to one vocabulary entry."""
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("es") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    if word.endswith("ing") and len(word) > 6:
        return word[:-3]
    return word


def _tokenize(text):
    return _TOKEN_RE.findall(text.lower())


def _clean_tokens(text):
    """Lowercase, tokenize, stem and drop stopwords / domain stopwords."""
    out = []
    for t in _tokenize(text):
        s = simple_stem(t)
        if len(s) < 2:
            continue
        if s in PROTECTED:
            out.append(s)
            continue
        if s in sk_text.ENGLISH_STOP_WORDS or s in DOMAIN_STOPWORDS:
            continue
        out.append(s)
    return out


# --------------------------------------------------------------------------
# Knowledge base (30 topics). Add/remove freely - the model refits on startup.
# --------------------------------------------------------------------------
INTENTS = {
    "best_time_to_visit": {
        "examples": [
            "when is the best time to visit mauritius",
            "best season to travel to mauritius",
            "when should i go to mauritius",
            "which months are best for a holiday here",
            "is december a good month to visit",
        ],
        "response": ("Mauritius is warm year-round. The driest, sunniest months are May to "
                     "December; January to March is hotter and more humid, and is the main "
                     "cyclone season."),
    },
    "attractions": {
        "examples": [
            "what are the best attractions in mauritius",
            "things to do in mauritius",
            "tourist spots in mauritius",
            "what sights shouldn't i miss",
            "top places to visit here",
        ],
        "response": ("Popular attractions include Le Morne Brabant, Chamarel Seven Coloured "
                     "Earths, Black River Gorges National Park, Ile aux Cerfs, and the capital "
                     "Port Louis with its waterfront and Central Market."),
    },
    "beaches": {
        "examples": [
            "what are the best beaches in mauritius",
            "nice beaches to relax on",
            "which beach is good for swimming",
            "are there safe beaches for kids",
            "best white sand beaches",
        ],
        "response": ("Favourites include Belle Mare and Flic en Flac (long white sands), Trou "
                     "aux Biches, Blue Bay (great for swimming), and Le Morne with its calm "
                     "lagoon. Many beaches are safe and sheltered for families."),
    },
    "transport": {
        "examples": [
            "how do i get around mauritius",
            "is there public transport in mauritius",
            "should i rent a car in mauritius",
            "whats the best way to travel around the island",
            "how do tourists usually get around the island",
        ],
        "response": ("Getting around is usually by rental car, taxi, or the public bus network, "
                     "which covers most of the island cheaply but can be slow. Ride-hailing apps "
                     "have limited coverage compared to taxis and hotel transfers."),
    },
    "accommodation": {
        "examples": [
            "where should i stay in mauritius",
            "hotel recommendations mauritius",
            "villa or hotel in mauritius",
            "what are good places to stay",
            "where do most tourists stay",
        ],
        "response": ("Mauritius offers beach resort hotels, private villas, serviced apartments, "
                     "and guesthouses. Villas and guesthouses have grown a lot in recent years as "
                     "an alternative to the big resorts."),
    },
    "currency": {
        "examples": [
            "what currency is used in mauritius",
            "do i need to exchange money",
            "can i pay in euros in mauritius",
            "should i bring cash or card",
            "are dollars accepted",
        ],
        "response": ("The local currency is the Mauritian Rupee (MUR). Major hotels and larger "
                     "shops often accept euros, US dollars, or cards, but carry some local "
                     "currency for taxis, markets, and smaller vendors."),
    },
    "weather": {
        "examples": [
            "what is the weather like in mauritius",
            "is it rainy in mauritius",
            "mauritius temperature",
            "how hot does it get",
            "does it rain a lot here",
        ],
        "response": ("Mauritius has a tropical climate: warm and relatively dry from May to "
                     "December, hotter, more humid, and wetter from January to March, which is "
                     "also cyclone season."),
    },
    "recent_cyclones": {
        "examples": [
            "when was the last cyclone in mauritius",
            "is there a cyclone coming",
            "are cyclones common in mauritius",
            "what happens during cyclone season",
        ],
        "response": ("Cyclones occur during the January-March season, though not every year makes "
                     "landfall. Warnings are issued in stages and resorts follow strict safety "
                     "plans. For current, dated storm information please check official weather "
                     "advisories."),
    },
    "food": {
        "examples": [
            "what food should i try in mauritius",
            "local cuisine mauritius",
            "what do people eat in mauritius",
            "whats traditional mauritian food",
            "is mauritian food spicy",
        ],
        "response": ("Mauritian cuisine blends Indian, Creole, Chinese, and French influences - "
                     "try dholl puri, rougaille, and biryani, along with fresh seafood along the "
                     "coast."),
    },
    "restaurants": {
        "examples": [
            "where should i eat in mauritius",
            "recommend a good restaurant",
            "best local restaurants near the beach",
            "where can i get good seafood",
            "any fine dining options",
        ],
        "response": ("You'll find everything from local street stalls to beachfront seafood grills "
                     "and fine-dining resort restaurants. Grand Baie and Flic en Flac have a good "
                     "concentration of options."),
    },
    "language": {
        "examples": [
            "what language is spoken in mauritius",
            "do people speak english in mauritius",
            "what languages do locals speak",
            "is french commonly spoken",
            "whats the main language on the island",
        ],
        "response": ("English is the official language and widely used, while French and Mauritian "
                     "Creole are most common day to day. As a tourist, English is generally "
                     "sufficient."),
    },
    "culture": {
        "examples": [
            "what is the culture like in mauritius",
            "what are mauritian traditions",
            "what festivals do mauritians celebrate",
            "is there a sega dance",
        ],
        "response": ("Mauritian culture blends Indian, Creole, Chinese, and European traditions. "
                     "The Sega dance and music, shared festivals, and a famously multicultural, "
                     "welcoming society are central to island life."),
    },
    "history": {
        "examples": [
            "what is the history of mauritius",
            "who colonised mauritius",
            "what is the aapravasi ghat",
            "when did mauritius become independent",
        ],
        "response": ("Mauritius was shaped by Dutch, French, and British rule before independence "
                     "in 1968. Historic sites like Aapravasi Ghat, Le Morne, and Chateau de "
                     "Labourdonnais reflect its plantation and multicultural past."),
    },
    "nature_wildlife": {
        "examples": [
            "what wildlife can i see in mauritius",
            "are there endemic birds in mauritius",
            "where can i see giant tortoises",
            "can i see dolphins or whales",
        ],
        "response": ("Mauritius is home to endemic species like the pink pigeon and echo "
                     "parakeet, plus giant tortoises at places like La Vanille and Casela. Black "
                     "River Gorges is the main national park. Dolphin and whale watching is "
                     "popular along the west coast."),
    },
    "hiking": {
        "examples": [
            "are there good hikes in mauritius",
            "can i climb le morne",
            "what are the best hiking trails",
            "is there a hike to a waterfall",
        ],
        "response": ("Black River Gorges and the Le Morne Brabant and Lion Mountain trails are "
                     "popular hikes. Bring water, go early for cooler conditions, and take a local "
                     "guide on harder routes like Le Morne."),
    },
    "water_sports": {
        "examples": [
            "what water sports can i do in mauritius",
            "is kite surfing popular in mauritius",
            "can i go sailing or paddleboarding",
            "is it good for surfing",
        ],
        "response": ("Lagoon watersports like kayaking, paddleboarding, and snorkelling are easy "
                     "to find, while Le Morne is world-famous for kite surfing. Sailing, "
                     "water-skiing, and parasailing are also available."),
    },
    "diving": {
        "examples": [
            "is mauritius good for diving",
            "where can i snorkel in mauritius",
            "best diving spots on the island",
            "is blue bay good for snorkelling",
        ],
        "response": ("Blue Bay Marine Park is excellent for snorkelling, and dive sites around "
                     "Flic en Flac and the west coast offer reefs, wrecks, and marine life. Many "
                     "operators offer beginner courses."),
    },
    "activities": {
        "examples": [
            "what activities can i do in mauritius",
            "fun things to do for couples",
            "are there adventure activities",
            "can i go on a catamaran cruise",
        ],
        "response": ("Popular activities range from catamaran cruises to Ile aux Cerfs and "
                     "southwest island tours, to adventure parks with zip lines, quad biking, and "
                     "waterfall visits. Most hotels can book excursions for you."),
    },
    "family_activities": {
        "examples": [
            "what activities are good for families",
            "things to do with kids in mauritius",
            "is mauritius family friendly",
            "are there safe beaches for children",
        ],
        "response": ("Mauritius is very family friendly - calm lagoon beaches, animal parks like "
                     "Casela and La Vanille, the museums in Port Louis, and short island boat "
                     "trips."),
    },
    "itineraries": {
        "examples": [
            "can you create a 7 day itinerary",
            "what should i do in 5 days in mauritius",
            "plan a two week trip for me",
            "suggest a day by day plan",
        ],
        "response": ("A common plan mixes beaches and exploration - for example: days 1-2 "
                     "relaxing in the north or east, a day at Ile aux Cerfs, a southwest tour "
                     "(Chamarel, Black River Gorges, Le Morne), Port Louis culture, and a final "
                     "beach day."),
    },
    "budget_travel": {
        "examples": [
            "is mauritius expensive",
            "what can i do on a budget",
            "is mauritius affordable for backpackers",
            "are there cheap places to stay",
        ],
        "response": ("Mauritius can be done on a tighter budget by staying in guesthouses, eating "
                     "at local eateries and street stalls, using public buses, and choosing free "
                     "or cheap activities like hiking and public beaches."),
    },
    "luxury": {
        "examples": [
            "what are the best luxury resorts",
            "is there a five star hotel in mauritius",
            "what is the most exclusive resort",
            "luxury experiences in mauritius",
        ],
        "response": ("Mauritius is known for upscale resorts, especially along the east and "
                     "northwest coasts, with private villas, spas, gourmet dining, and top "
                     "service."),
    },
    "honeymoon": {
        "examples": [
            "is mauritius good for a honeymoon",
            "romantic things to do in mauritius",
            "best hotels for couples",
            "romantic dinner ideas on the beach",
        ],
        "response": ("Mauritius is a classic honeymoon destination - private beach villas, sunset "
                     "catamaran cruises, candlelit beach dinners, and couples spa treatments. Many "
                     "resorts offer dedicated honeymoon packages."),
    },
    "safety": {
        "examples": [
            "is mauritius safe for tourists",
            "is mauritius safe at night",
            "is the water safe to drink",
            "do i need any vaccinations",
        ],
        "response": ("Mauritius is generally a very safe destination, with low violent crime and a "
                     "welcoming population. Take normal precautions at night, stick to bottled "
                     "water, and check current travel-health advice before you go."),
    },
    "visa": {
        "examples": [
            "do i need a visa for mauritius",
            "visa requirements for mauritius",
            "how long can tourists stay",
            "is mauritius visa free for my country",
        ],
        "response": ("Many nationalities can enter visa-free as tourists for up to 90 days. Rules "
                     "change, so confirm the latest entry requirements with an official source "
                     "before travelling."),
    },
    "shopping": {
        "examples": [
            "what can i buy in mauritius",
            "where can i shop for souvenirs",
            "best shopping malls in mauritius",
            "where can i buy local rum",
        ],
        "response": ("Great souvenirs include Mauritian rum, vanilla, model ships, handcrafts, and "
                     "textiles. The Port Louis Central Market and craft markets are good for local "
                     "items."),
    },
    "nightlife": {
        "examples": [
            "what is the nightlife like in mauritius",
            "best bars and clubs in mauritius",
            "nightlife in grand baie",
            "what to do in the evening",
        ],
        "response": ("Grand Baie is the liveliest nightlife spot, with bars, restaurants, and "
                     "clubs, and a number of hotels host live music and themed evenings. Elsewhere "
                     "the evening is quieter."),
    },
    "greetings": {
        "examples": [
            "hello",
            "hi there",
            "good morning",
            "hey how are you",
            "bonjour",
        ],
        "response": ("Bonjour! I'm your Mauritius tourism assistant. Ask me about beaches, food, "
                     "where to stay, getting around, and activities - or pick a suggested question "
                     "to get started."),
    },
    "thanks": {
        "examples": [
            "thank you",
            "thanks a lot",
            "that helps",
            "great thanks",
        ],
        "response": ("You're very welcome! If you have any more questions about planning your trip "
                     "to Mauritius, just ask."),
    },
}

FALLBACK = (
    "Hmm, I couldn't confidently match that in my local knowledge base. "
    "Please try rephrasing your question, or the AI assistant will take "
    "over if it is available."
)


# --------------------------------------------------------------------------
# Chatbot class - exposes the interface used by streamlit_app.py
# --------------------------------------------------------------------------
class TourismChatbot:
    """Intent classifier using TF-IDF + cosine similarity with a confidence
    threshold and a margin requirement over the runner-up intent."""

    def __init__(self):
        self._fit()

    def _fit(self):
        # Collect every example phrase and remember its intent.
        self._examples = []
        self._example_intent = {}
        for name, intent in INTENTS.items():
            for ex in intent["examples"]:
                self._examples.append(ex)
                self._example_intent[ex] = name

        # Build the vocabulary from the example phrases.
        vocab = set()
        for ex in self._examples:
            vocab.update(_clean_tokens(ex))
        self._vocab = sorted(vocab)

        # Tokenizer that also fuzzy-corrects typos against the vocabulary.
        def tokenizer(text):
            corrected = []
            for t in _clean_tokens(text):
                if t in self._vocab:
                    corrected.append(t)
                else:
                    match = difflib.get_close_matches(t, self._vocab, n=1, cutoff=0.55)
                    corrected.append(match[0] if match else t)
            return corrected

        self._vectorizer = TfidfVectorizer(
            vocabulary=self._vocab,
            tokenizer=tokenizer,
            preprocessor=lambda x: x,
            token_pattern=None,
            lowercase=False,
        )
        self._X = self._vectorizer.fit_transform(self._examples)

    def _score_intents(self, question):
        q = self._vectorizer.transform([question])
        sims = cosine_similarity(q, self._X)[0]
        scores = {}
        for i, ex in enumerate(self._examples):
            name = self._example_intent[ex]
            # Aggregate by intent: take the best-matching example per intent.
            scores[name] = max(scores.get(name, 0.0), sims[i])
        return scores

    def respond_debug(self, question):
        """Return a dict with is_confident, the matched intent, score, the
        runner-up, and the canonical response if confident."""
        scores = self._score_intents(question)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_name, top_score = ranked[0]
        runner_name, runner_score = ranked[1] if len(ranked) > 1 else (None, 0.0)

        confident = top_score >= MIN_SCORE and (top_score - runner_score) >= MIN_MARGIN

        return {
            "is_confident": confident,
            "intent": top_name if confident else None,
            "score": top_score,
            "runner_up": runner_name,
            "runner_up_score": runner_score,
            "response": INTENTS[top_name]["response"] if confident else "",
        }

    def respond(self, question):
        """Convenience wrapper returning (text, source)."""
        debug = self.respond_debug(question)
        if debug["is_confident"]:
            return debug["response"], "knowledge_base"
        return FALLBACK, "fallback"
