"""
Mauritius Tourism Chatbot Core
------------------------------
Hybrid intent-based tourism chatbot.

Features:
1. TF-IDF local tourism knowledge base
2. 30+ tourism intents
3. Fuzzy typo correction
4. Intent confidence + margin checking
5. Dedicated beach intent
6. Complex/personalised-question detection
7. Gemini fallback handled by streamlit_app.py / llm_fallback.py

The local layer answers common tourism questions quickly.
Complex, personalised and multi-step questions are passed to Gemini.
"""

import re
import difflib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text as sk_text
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# TOKENISATION
# ============================================================

_TOKEN_RE = re.compile(r"[a-zA-Z]{2,}")


def simple_stem(word):
    """Small dependency-free stemmer."""

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


# ============================================================
# TOURISM INTENTS
# ============================================================

INTENTS = {

    # --------------------------------------------------------
    # BEACHES
    # --------------------------------------------------------

    "beaches": {
        "examples": [
            "what are the best beaches in mauritius",
            "best beaches in mauritius",
            "which beaches should i visit",
            "what beaches do you recommend",
            "where are the nicest beaches",
            "which beach is the most beautiful",
            "best beach for swimming",
            "best beach for families",
            "where can i find beautiful beaches",
            "which beaches are worth visiting",
            "top beaches to visit",
            "where are the best beaches",
            "recommend beaches in mauritius",
        ],
        "response": (
            "Mauritius has many beautiful beaches, each with a different "
            "character. Trou aux Biches is popular for calm water and "
            "snorkelling, Belle Mare is known for its long white-sand beach, "
            "Flic en Flac is popular for sunsets and swimming, and Blue Bay "
            "is well known for its marine environment and clear water. "
            "For a quieter experience, consider beaches along the south "
            "and south-east coasts."
        ),
    },

    # --------------------------------------------------------
    # BEST TIME
    # --------------------------------------------------------

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
        "response": (
            "Mauritius is warm year-round. The driest, sunniest months are "
            "May to December; January to March is hotter and more humid, "
            "and is the main cyclone season."
        ),
    },

    # --------------------------------------------------------
    # ATTRACTIONS
    # --------------------------------------------------------

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
        "response": (
            "Popular attractions include Le Morne Brabant, Chamarel Seven "
            "Coloured Earths, Black River Gorges National Park, Ile aux "
            "Cerfs, and the capital Port Louis with its waterfront and "
            "Central Market."
        ),
    },

    # --------------------------------------------------------
    # TRANSPORT
    # --------------------------------------------------------

    "transport": {
        "examples": [
            "how do i get around mauritius",
            "is there public transport in mauritius",
            "should i rent a car in mauritius",
            "what's the best way to travel around the island",
            "can i use uber in mauritius",
            "do buses run frequently",
            "is it easy to drive around",
            "how do tourists usually get around the island",
            "what transportation options are available",
            "how does the bus and taxi system work",
        ],
        "response": (
            "Getting around is usually by rental car, taxi, or the public "
            "bus network, which covers most of the island cheaply but can "
            "be slow. Ride-hailing apps have limited coverage compared to "
            "taxis and hotel transfers."
        ),
    },

    # --------------------------------------------------------
    # ACCOMMODATION
    # --------------------------------------------------------

    "accommodation": {
        "examples": [
            "where should i stay in mauritius",
            "hotel recommendations mauritius",
            "villa or hotel in mauritius",
            "what are good places to stay",
            "any resort recommendations",
            "should i book an airbnb",
            "best areas for hotels",
            "where do most tourists stay",
            "what accommodation options are available",
            "is self-catering accommodation available",
            "can you recommend a good hotel or resort",
        ],
        "response": (
            "Mauritius offers beach resort hotels, private villas, "
            "serviced apartments, and guesthouses. Villas and guesthouses "
            "are popular alternatives to large resorts."
        ),
    },

    # --------------------------------------------------------
    # CURRENCY
    # --------------------------------------------------------

    "currency": {
        "examples": [
            "what currency is used in mauritius",
            "do i need to exchange money",
            "can i pay in euros in mauritius",
            "what money do they use here",
            "should i bring cash or card",
            "are dollars accepted",
            "where can i exchange money",
            "is it a cash-based economy",
        ],
        "response": (
            "The local currency is the Mauritian Rupee (MUR). Major hotels "
            "and larger shops often accept cards and sometimes major foreign "
            "currencies, but carrying some local currency is useful for taxis, "
            "markets and smaller vendors."
        ),
    },

    # --------------------------------------------------------
    # WEATHER
    # --------------------------------------------------------

    "weather": {
        "examples": [
            "what is the weather like in mauritius",
            "is it rainy in mauritius",
            "mauritius temperature",
            "how hot does it get",
            "does it rain a lot here",
            "what's the climate like on the island",
            "is it humid",
            "when is cyclone season",
        ],
        "response": (
            "Mauritius has a tropical climate. The period from May to "
            "December is generally drier, while January to March is hotter, "
            "more humid and wetter, and is also the main cyclone period."
        ),
    },

    # --------------------------------------------------------
    # FOOD
    # --------------------------------------------------------

    "food": {
        "examples": [
            "what food should i try in mauritius",
            "local cuisine mauritius",
            "what do people eat in mauritius",
            "what's traditional mauritian food",
            "any must-try dishes",
            "what's the local cuisine like",
            "is mauritian food spicy",
            "what should i eat while i'm there",
        ],
        "response": (
            "Mauritian cuisine blends Indian, Creole, Chinese and French "
            "influences. Try dholl puri, rougaille and biryani, along with "
            "fresh seafood and local street food."
        ),
    },

    # --------------------------------------------------------
    # LANGUAGE
    # --------------------------------------------------------

    "language": {
        "examples": [
            "what language is spoken in mauritius",
            "do people speak english in mauritius",
            "what languages do locals speak",
            "will i have trouble communicating in english",
            "is french commonly spoken",
            "do mauritians speak creole",
            "what's the main language on the island",
        ],
        "response": (
            "English is widely used in government, education and tourism. "
            "French is also widely spoken, while Mauritian Creole is the "
            "language most commonly used in everyday conversation."
        ),
    },

    # --------------------------------------------------------
    # VISA
    # --------------------------------------------------------

    "visa": {
        "examples": [
            "do i need a visa to visit mauritius",
            "how long can i stay without a visa",
            "visa requirements for mauritius",
            "do i need a visa on arrival",
            "how many days can i stay visa free",
            "what documents do i need to enter mauritius",
            "is a passport enough to enter mauritius",
        ],
        "response": (
            "Visa requirements depend on nationality and the purpose and "
            "length of the visit. Travellers should check the latest official "
            "Mauritian immigration guidance before travelling, as requirements "
            "can change."
        ),
    },

    # --------------------------------------------------------
    # SAFETY
    # --------------------------------------------------------

    "safety": {
        "examples": [
            "is mauritius safe for tourists",
            "is it safe to walk around at night",
            "how safe is mauritius",
            "is crime a problem in mauritius",
            "should i be worried about safety",
            "any safety tips for visiting mauritius",
        ],
        "response": (
            "Mauritius is generally considered a safe destination for "
            "tourists. As anywhere, use normal precautions with valuables, "
            "particularly on beaches, at night and in crowded areas."
        ),
    },

    # --------------------------------------------------------
    # TIPPING
    # --------------------------------------------------------

    "tipping": {
        "examples": [
            "do you tip in mauritius",
            "is tipping expected in mauritius",
            "how much should i tip",
            "do i need to tip taxi drivers",
            "is a service charge included at restaurants",
        ],
        "response": (
            "Tipping is not obligatory but is appreciated. Some hotels and "
            "restaurants already include a service charge. A small tip for "
            "good service is generally appropriate."
        ),
    },

    # --------------------------------------------------------
    # CONNECTIVITY
    # --------------------------------------------------------

    "connectivity": {
        "examples": [
            "where can i buy a sim card in mauritius",
            "is there good wifi in mauritius",
            "does my phone work in mauritius",
            "how do i get internet access on the island",
            "can i use mobile data in mauritius",
            "is there roaming coverage in mauritius",
        ],
        "response": (
            "Local prepaid SIM cards are available at the airport and in "
            "towns. Most hotels, resorts and many cafes also offer Wi-Fi. "
            "Check roaming charges with your home mobile provider before "
            "travelling."
        ),
    },

    # --------------------------------------------------------
    # BUDGET
    # --------------------------------------------------------

    "budget": {
        "examples": [
            "how much does a week in mauritius cost",
            "is mauritius an expensive destination",
            "what's a typical daily budget for mauritius",
            "is mauritius cheap or expensive",
            "how much money should i bring",
        ],
        "response": (
            "Costs vary considerably. Budget guesthouses and local food can "
            "make a trip relatively affordable, while luxury resorts, "
            "activities and imported goods can increase costs considerably."
        ),
    },

    # --------------------------------------------------------
    # ELECTRICITY
    # --------------------------------------------------------

    "electricity": {
        "examples": [
            "what plug type is used in mauritius",
            "what voltage does mauritius use",
            "do i need an adapter for mauritius",
            "what socket type is used there",
        ],
        "response": (
            "Mauritius generally uses UK-style Type G three-pin sockets at "
            "230V and 50Hz. Visitors with different plug types may need a "
            "travel adapter."
        ),
    },

    # --------------------------------------------------------
    # HEALTH
    # --------------------------------------------------------

    "health": {
        "examples": [
            "do i need vaccinations to travel to mauritius",
            "is malaria a risk in mauritius",
            "are there health risks in mauritius",
            "do i need travel insurance for mauritius",
            "what medical facilities are available there",
        ],
        "response": (
            "Travellers should ensure routine vaccinations are up to date "
            "and check current health guidance before travelling. Mauritius "
            "has medical facilities available for tourists, and travel "
            "insurance covering medical care is recommended."
        ),
    },

    # --------------------------------------------------------
    # DIVING / SNORKELLING
    # --------------------------------------------------------

    "diving_snorkeling": {
        "examples": [
            "where's the best place to go scuba diving in mauritius",
            "is snorkeling good in mauritius",
            "best dive sites in mauritius",
            "can beginners try diving in mauritius",
            "where can i go snorkelling",
        ],
        "response": (
            "Mauritius has good diving and snorkelling around its lagoons "
            "and reefs. Popular areas include Flic en Flac, Blue Bay Marine "
            "Park and parts of the north coast. Dive centres offer trips "
            "and courses for beginners and experienced divers."
        ),
    },

    # --------------------------------------------------------
    # HONEYMOON
    # --------------------------------------------------------

    "honeymoon": {
        "examples": [
            "is mauritius good for honeymoons",
            "is mauritius romantic for couples",
            "best honeymoon resorts in mauritius",
            "is mauritius a good place for a romantic getaway",
        ],
        "response": (
            "Mauritius is a popular honeymoon destination, with resorts "
            "offering couples' packages, private dinners and romantic "
            "experiences. Belle Mare, Trou aux Biches and the south-west "
            "coast are popular areas for couples."
        ),
    },

    # --------------------------------------------------------
    # FAMILY
    # --------------------------------------------------------

    "family_kids": {
        "examples": [
            "what activities are good for kids in mauritius",
            "is mauritius family friendly",
            "things to do with children in mauritius",
            "are there family resorts in mauritius",
            "what can families do in mauritius",
        ],
        "response": (
            "Mauritius is very family friendly. Options include calm "
            "lagoon swimming, glass-bottom boat trips, nature attractions, "
            "wildlife experiences and family-oriented resorts."
        ),
    },

    # --------------------------------------------------------
    # FLIGHTS
    # --------------------------------------------------------

    "flights": {
        "examples": [
            "how long is the flight to mauritius",
            "which airlines fly to mauritius",
            "is there a direct flight to mauritius",
            "how far is mauritius by plane",
        ],
        "response": (
            "Flight time depends on the departure location. Mauritius is "
            "served by Sir Seewoosagur Ramgoolam International Airport (MRU) "
            "and Air Mauritius and other international airlines operate "
            "routes to the island."
        ),
    },

    # --------------------------------------------------------
    # AIRPORT TRANSFER
    # --------------------------------------------------------

    "airport_transfer": {
        "examples": [
            "how far is the airport from the main hotels",
            "how do i get from the airport to my hotel",
            "is airport transfer included with resorts",
            "how long is the drive from the airport",
        ],
        "response": (
            "The international airport is in the south-east of Mauritius. "
            "Transfer times depend on your destination, with the north and "
            "west generally taking longer than the south-east. Hotels can "
            "usually arrange transfers and taxis are available at the airport."
        ),
    },

    # --------------------------------------------------------
    # REGIONS
    # --------------------------------------------------------

    "regions": {
        "examples": [
            "which part of mauritius should i stay in",
            "best area to stay in mauritius",
            "north or south mauritius for a holiday",
            "which coast is best in mauritius",
        ],
        "response": (
            "The north is lively and offers shopping and nightlife. The west "
            "is popular for sunsets and beaches. The east is quieter and is "
            "known for long beaches, while the south is more rugged and "
            "nature-focused."
        ),
    },

    # --------------------------------------------------------
    # NIGHTLIFE
    # --------------------------------------------------------

    "nightlife": {
        "examples": [
            "what's the nightlife like in mauritius",
            "are there bars and clubs in mauritius",
            "where can i go out at night in mauritius",
        ],
        "response": (
            "Nightlife is particularly concentrated around Grand Baie in "
            "the north, with bars and clubs. Other parts of the island tend "
            "to have quieter evenings centred around restaurants and hotels."
        ),
    },

    # --------------------------------------------------------
    # SHOPPING
    # --------------------------------------------------------

    "shopping": {
        "examples": [
            "where can i go shopping in mauritius",
            "best markets in mauritius",
            "what souvenirs should i buy in mauritius",
        ],
        "response": (
            "Port Louis Central Market and Caudan Waterfront are popular "
            "for souvenirs, textiles and local products. Shopping malls "
            "such as Bagatelle and Grand Baie La Croisette offer a more "
            "modern shopping experience."
        ),
    },

    # --------------------------------------------------------
    # DRIVING
    # --------------------------------------------------------

    "driving_license": {
        "examples": [
            "can i drive with a foreign driving license in mauritius",
            "do i need an international driving permit for mauritius",
            "what's needed to rent a car in mauritius",
        ],
        "response": (
            "Visitors can generally drive in Mauritius using a valid foreign "
            "driving licence subject to applicable rules. Rental companies "
            "may have their own requirements. Mauritius drives on the left."
        ),
    },

    # --------------------------------------------------------
    # DAY TRIPS
    # --------------------------------------------------------

    "day_trips": {
        "examples": [
            "can i visit rodrigues island as a day trip",
            "day trips from mauritius",
            "can i visit other islands near mauritius",
        ],
        "response": (
            "Popular day-trip options include Ile aux Cerfs, Ile aux "
            "Benitiers and catamaran cruises. Rodrigues is usually better "
            "suited to a multi-day visit rather than a day trip."
        ),
    },

    # --------------------------------------------------------
    # LGBTQ
    # --------------------------------------------------------

    "lgbtq": {
        "examples": [
            "is mauritius lgbtq friendly",
            "is it safe to travel to mauritius as a gay couple",
            "lgbtq travel mauritius",
        ],
        "response": (
            "Tourist areas and resorts are generally welcoming, although "
            "social attitudes can be more conservative than in some Western "
            "destinations. Travellers should be mindful of local social norms."
        ),
    },

    # --------------------------------------------------------
    # SOLO FEMALE
    # --------------------------------------------------------

    "solo_female": {
        "examples": [
            "is mauritius safe for solo female travelers",
            "can a woman travel alone in mauritius",
            "solo female travel mauritius safety",
        ],
        "response": (
            "Many women travel solo in Mauritius. Tourist areas are generally "
            "safe, but normal precautions are recommended, particularly "
            "around isolated areas at night."
        ),
    },

    # --------------------------------------------------------
    # WEDDINGS
    # --------------------------------------------------------

    "weddings": {
        "examples": [
            "can foreigners get married in mauritius",
            "how do i organize a wedding in mauritius",
            "is mauritius good for destination weddings",
        ],
        "response": (
            "Mauritius is a popular destination-wedding location. Foreign "
            "couples can marry there, subject to documentation and legal "
            "requirements. Resorts and wedding planners can often assist "
            "with the arrangements."
        ),
    },

    # --------------------------------------------------------
    # DUTY FREE
    # --------------------------------------------------------

    "duty_free": {
        "examples": [
            "what's the duty free allowance in mauritius",
            "how much alcohol can i bring into mauritius",
            "customs allowance for mauritius",
        ],
        "response": (
            "Mauritius has duty-free allowances for items such as alcohol, "
            "tobacco and perfume. Limits can change, so travellers should "
            "check the latest customs guidance before travelling."
        ),
    },

    # --------------------------------------------------------
    # TAP WATER
    # --------------------------------------------------------

    "tap_water": {
        "examples": [
            "can you drink tap water in mauritius",
            "is tap water safe in mauritius",
            "should i drink bottled water in mauritius",
        ],
        "response": (
            "Tap water is treated and generally considered safe in many "
            "tourist areas, although some visitors prefer bottled water."
        ),
    },

    # --------------------------------------------------------
    # INSECTS
    # --------------------------------------------------------

    "insects": {
        "examples": [
            "are there mosquitoes in mauritius",
            "is dengue a risk in mauritius",
            "should i bring mosquito repellent to mauritius",
        ],
        "response": (
            "Mosquitoes are present, particularly during wetter periods. "
            "Using insect repellent and covering exposed skin in the evening "
            "are sensible precautions."
        ),
    },

    # --------------------------------------------------------
    # PUBLIC HOLIDAYS
    # --------------------------------------------------------

    "public_holidays": {
        "examples": [
            "what are the public holidays in mauritius",
            "is mauritius closed on public holidays",
            "when are the holidays in mauritius",
        ],
        "response": (
            "Mauritius celebrates a range of public holidays reflecting its "
            "multicultural population, including New Year, Independence Day, "
            "Diwali, Eid and Christmas. Some businesses may operate reduced "
            "hours on public holidays."
        ),
    },

    # --------------------------------------------------------
    # ENTRY REQUIREMENTS
    # --------------------------------------------------------

    "entry_requirements": {
        "examples": [
            "what are the current entry rules for mauritius",
            "do i need any health documents to enter mauritius",
            "what documents do i need to enter mauritius",
        ],
        "response": (
            "Entry requirements can change according to traveller nationality "
            "and current regulations. Travellers should check the latest "
            "official Mauritian immigration guidance before departure."
        ),
    },

    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    "history": {
        "examples": [
            "what is the history of mauritius",
            "how was mauritius colonized",
            "who discovered mauritius",
            "when did mauritius gain independence",
            "was mauritius a british colony",
            "brief history of mauritius",
            "tell me about mauritius history",
        ],
        "response": (
            "Mauritius was settled by European powers during the colonial "
            "period, first by the Dutch and later by the French and British. "
            "Mauritius gained independence on 12 March 1968 and became a "
            "republic in 1992."
        ),
    },

    # --------------------------------------------------------
    # POPULATION
    # --------------------------------------------------------

    "population": {
        "examples": [
            "what is the population of mauritius",
            "how many people live in mauritius",
            "how big is the population there",
            "population of mauritius",
            "how many people live on the island",
        ],
        "response": (
            "Mauritius has a population of around 1.3 million people and is "
            "one of the more densely populated island countries in Africa."
        ),
    },

    # --------------------------------------------------------
    # GEOGRAPHY
    # --------------------------------------------------------

    "geography": {
        "examples": [
            "where is mauritius located",
            "how big is mauritius",
            "what continent is mauritius part of",
            "is mauritius near madagascar",
            "geography of mauritius",
            "how far is mauritius from africa",
        ],
        "response": (
            "Mauritius is a volcanic island in the Indian Ocean, around "
            "2,000 kilometres from the south-east coast of Africa and east "
            "of Madagascar. It forms part of the Mascarene Islands."
        ),
    },

    # --------------------------------------------------------
    # GOVERNMENT
    # --------------------------------------------------------

    "government": {
        "examples": [
            "what type of government does mauritius have",
            "is mauritius a democracy",
            "who governs mauritius",
            "is mauritius a republic",
            "what is the political system in mauritius",
        ],
        "response": (
            "Mauritius is a parliamentary republic. It has a President as "
            "head of state and a Prime Minister who leads the government."
        ),
    },

    # --------------------------------------------------------
    # ECONOMY
    # --------------------------------------------------------

    "economy": {
        "examples": [
            "what is the economy of mauritius based on",
            "what industries does mauritius have",
            "how does mauritius make money",
            "is mauritius a rich country",
            "what is mauritius known for economically",
        ],
        "response": (
            "Mauritius has a diversified economy involving tourism, financial "
            "services, textiles and manufacturing, sugar production, and "
            "information and communication technology."
        ),
    },

    # --------------------------------------------------------
    # CYCLONES
    # --------------------------------------------------------

    "recent_cyclones": {
        "examples": [
            "when was the last cyclone in mauritius",
            "has a cyclone hit mauritius recently",
            "was there a cyclone this year in mauritius",
            "did a cyclone hit mauritius recently",
            "what was the most recent cyclone in mauritius",
            "latest cyclone news mauritius",
            "any recent cyclone warnings for mauritius",
            "how bad was the last cyclone season",
        ],
        "response": (
            "Cyclone activity changes each season, so this should not be "
            "treated as live information. For current cyclone warnings and "
            "official weather information, travellers should check the "
            "Mauritius Meteorological Services before travelling, especially "
            "during cyclone season."
        ),
    },
}


# ============================================================
# STOPWORDS
# ============================================================

DOMAIN_STOPWORDS = {
    "mauritius",
    "mauritian",
    "mauritians",
    "island",
    "here",
    "there",
    "tell",
    "know",
    "want",
    "explain",
}

STOPWORDS = list(
    sk_text.ENGLISH_STOP_WORDS.union(DOMAIN_STOPWORDS)
)


# ============================================================
# FALLBACK
# ============================================================

FALLBACK = (
    "I'm not fully sure about that one. I can help with topics such as "
    "beaches, attractions, activities, food, weather, accommodation, "
    "transport, safety, visas, budgeting and more. You can also ask me "
    "to create a personalised Mauritius itinerary."
)


# ============================================================
# MATCHING THRESHOLDS
# ============================================================

MIN_SCORE = 0.22
MIN_MARGIN = 0.04


# ============================================================
# CHATBOT CLASS
# ============================================================

class TourismChatbot:

    def __init__(self, intents=INTENTS):

        self.intents = intents

        self.utterances = []
        self.labels = []

        for name, data in intents.items():

            for example in data["examples"]:

                self.utterances.append(example)
                self.labels.append(name)

        self._stopwords = set(STOPWORDS)

        self.vectorizer = TfidfVectorizer(
            analyzer=self._analyze
        )

        self.matrix = self.vectorizer.fit_transform(
            self.utterances
        )

        self.vocab = set(
            self.vectorizer.vocabulary_.keys()
        )

        self._stem_vocab = sorted(
            word
            for word in self.vocab
            if " " not in word
        )

    # ========================================================
    # TOKEN ANALYSER
    # ========================================================

    def _analyze(self, doc):

        tokens = [
            token
            for token in _tokenize(doc)
            if token not in self._stopwords
        ]

        stems = [
            simple_stem(token)
            for token in tokens
        ]

        bigrams = [
            f"{a} {b}"
            for a, b in zip(stems, stems[1:])
        ]

        return stems + bigrams

    # ========================================================
    # TYPO CORRECTION
    # ========================================================

    def _correct_typos(self, query):

        words = _tokenize(query)

        corrected = []

        for word in words:

            if word in self._stopwords or len(word) < 4:

                corrected.append(word)

                continue

            stem = simple_stem(word)

            if stem in self.vocab:

                corrected.append(word)

                continue

            match = difflib.get_close_matches(
                stem,
                self._stem_vocab,
                n=1,
                cutoff=0.90,
            )

            if (
                match
                and abs(len(match[0]) - len(stem)) <= 2
            ):

                corrected.append(match[0])

            else:

                corrected.append(word)

        return " ".join(corrected)

    # ========================================================
    # COMPLEX QUESTION DETECTION
    # ========================================================

    def _needs_llm(self, query):
        """
        Identify questions that should be handled by Gemini.

        These include:
        - itinerary requests
        - personalised recommendations
        - trip planning
        - multi-preference questions
        - requests requiring reasoning across several topics
        """

        q = query.lower().strip()

        # Explicit planning phrases
        planning_phrases = [

            "create a personalised itinerary",
            "create a personalized itinerary",

            "create an itinerary",
            "make an itinerary",

            "plan my trip",
            "plan a trip",
            "plan my holiday",
            "plan my vacation",

            "help me plan",
            "help plan my trip",

            "make me a plan",
            "make a travel plan",

            "personalised itinerary",
            "personalized itinerary",

            "personalised trip",
            "personalized trip",

            "personalised holiday",
            "personalized holiday",

            "recommend based on",
            "recommend something for me",

            "what would you recommend for me",

            "suggest an itinerary",

            "build an itinerary",
            "design an itinerary",
        ]

        if any(
            phrase in q
            for phrase in planning_phrases
        ):
            return True

        # Itinerary-related words
        itinerary_words = [
            "itinerary",
            "trip plan",
            "travel plan",
            "holiday plan",
        ]

        if any(
            phrase in q
            for phrase in itinerary_words
        ):
            return True

        # Multiple personal preferences
        preference_words = [
            "i enjoy",
            "i like",
            "i love",
            "my family",
            "with my children",
            "with kids",
            "with my partner",
            "as a couple",
            "i'm travelling",
            "i am travelling",
            "i'm traveling",
            "i am traveling",
        ]

        preference_count = sum(
            1
            for phrase in preference_words
            if phrase in q
        )

        # If the user describes their trip + preferences,
        # Gemini should create the response.
        if preference_count >= 1:

            tourism_preferences = [
                "beach",
                "beaches",
                "nature",
                "food",
                "culture",
                "adventure",
                "hiking",
                "shopping",
                "nightlife",
                "diving",
                "snorkelling",
                "snorkeling",
                "relax",
                "relaxing",
            ]

            preference_topics = sum(
                1
                for word in tourism_preferences
                if word in q
            )

            if preference_topics >= 2:
                return True

        return False

    # ========================================================
    # DEBUG / INTENT MATCHING
    # ========================================================

    def respond_debug(self, query):

        # ----------------------------------------------------
        # EMPTY INPUT
        # ----------------------------------------------------

        if not query or not query.strip():

            return {
                "matched_label": None,
                "score": 0.0,
                "margin": 0.0,
                "is_confident": False,
                "response": None,
                "ranked": [],
                "corrected_query": query,
                "needs_llm": False,
            }

        # ----------------------------------------------------
        # COMPLEX / PERSONALISED REQUEST
        # ----------------------------------------------------

        if self._needs_llm(query):

            return {
                "matched_label": None,
                "score": 0.0,
                "margin": 0.0,
                "is_confident": False,
                "response": None,
                "ranked": [],
                "corrected_query": query,
                "needs_llm": True,
            }

        # ----------------------------------------------------
        # LOCAL TF-IDF MATCHING
        # ----------------------------------------------------

        cleaned = self._correct_typos(query)

        transformed = self.vectorizer.transform(
            [cleaned]
        )

        sims = cosine_similarity(
            transformed,
            self.matrix
        )[0]

        # ----------------------------------------------------
        # BEST SCORE PER INTENT
        # ----------------------------------------------------

        best_per_label = {}

        for label, score in zip(
            self.labels,
            sims
        ):

            if score > best_per_label.get(
                label,
                -1
            ):

                best_per_label[label] = score

        ranked = sorted(
            best_per_label.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        # Safety check
        if not ranked:

            return {
                "matched_label": None,
                "score": 0.0,
                "margin": 0.0,
                "is_confident": False,
                "response": None,
                "ranked": [],
                "corrected_query": cleaned,
                "needs_llm": True,
            }

        top_label = ranked[0][0]
        top_score = ranked[0][1]

        runner_up_score = (
            ranked[1][1]
            if len(ranked) > 1
            else 0.0
        )

        margin = (
            top_score
            - runner_up_score
        )

        is_confident = (
            top_score >= MIN_SCORE
            and margin >= MIN_MARGIN
        )

        return {
            "matched_label": (
                top_label
                if is_confident
                else None
            ),

            "score": float(top_score),

            "margin": float(margin),

            "is_confident": is_confident,

            "response": (
                self.intents[top_label]["response"]
                if is_confident
                else None
            ),

            "ranked": [
                (label, float(score))
                for label, score in ranked[:3]
            ],

            "corrected_query": cleaned,

            "needs_llm": not is_confident,
        }

    # ========================================================
    # OFFLINE RESPONSE
    # ========================================================

    def respond(self, query):

        debug = self.respond_debug(query)

        if (
            debug["is_confident"]
            and debug["response"]
        ):

            return debug["response"]

        return FALLBACK


# ============================================================
# SIMPLE TERMINAL TEST
# ============================================================

if __name__ == "__main__":

    bot = TourismChatbot()

    print(
        "\nMauritius Tourism Chatbot\n"
        "Type 'quit' to exit.\n"
    )

    while True:

        question = input("You: ").strip()

        if question.lower() in {
            "quit",
            "exit",
        }:

            break

        result = bot.respond_debug(
            question
        )

        if result["needs_llm"]:

            print(
                "Routing to Gemini / LLM fallback..."
            )

        elif result["is_confident"]:

            print(
                "Bot:",
                result["response"]
            )

        else:

            print(
                "Bot:",
                FALLBACK
            )

        print()
