"""
Intent-based tourism chatbot prototype - v2.

Fixes found by the functional evaluation (Table 4.4 / Mauritius_Chatbot_
Evaluation_Report.docx) of the v1 prototype:
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
     "weather" topic, since "cyclone" appears in both - the fix keeps it as
     its own topic with an explicit, dated, honestly-caveated answer instead
     of a generic climate description.

Run:
    pip install scikit-learn --break-system-packages
    python chatbot_core.py
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
    etc. collapse to the same vocabulary entry instead of being treated as
    unrelated, unknown words."""
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
        "response": ("Getting around is usually by rental car, taxi, or the public "
                     "bus network, which covers most of the island cheaply but can "
                     "be slow. Ride-hailing apps have limited coverage compared to "
                     "taxis and hotel transfers."),
    },
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
        "response": ("Mauritius offers beach resort hotels, private villas, "
                     "serviced apartments, and guesthouses. Villas and "
                     "guesthouses have grown a lot in recent years as an "
                     "alternative to the big resorts."),
    },
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
        "response": ("The local currency is the Mauritian Rupee (MUR). Major "
                     "hotels and larger shops often accept euros, US dollars, or "
                     "cards, but carry some local currency for taxis, markets, "
                     "and smaller vendors."),
    },
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
        "response": ("Mauritius has a tropical climate: warm and relatively dry "
                     "from May to December, hotter, more humid, and wetter from "
                     "January to March, which is also cyclone season."),
    },
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
        "response": ("Mauritian cuisine blends Indian, Creole, Chinese, and French "
                     "influences - try dholl puri, rougaille, and biryani, along "
                     "with fresh seafood along the coast."),
    },
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
        "response": ("English is the official language and is used in schools, "
                     "government, and most tourism settings. French is also "
                     "widely spoken, and Mauritian Creole is the language most "
                     "people use day to day."),
    },
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
        "response": ("Many nationalities can enter Mauritius visa-free for tourism "
                     "for a limited period (commonly up to 60 days), with a valid "
                     "passport, proof of onward travel, and accommodation details. "
                     "Requirements vary by nationality and change over time, so "
                     "always confirm with the nearest Mauritian embassy or the "
                     "official passport and immigration office before you fly."),
    },
    "safety": {
        "examples": [
            "is mauritius safe for tourists",
            "is it safe to walk around at night",
            "how safe is mauritius",
            "is crime a problem in mauritius",
            "should i be worried about safety",
            "any safety tips for visiting mauritius",
        ],
        "response": ("Mauritius is generally considered safe for tourists, with "
                     "low rates of violent crime. As anywhere, use normal "
                     "precautions with valuables on beaches, at night, and in "
                     "crowded areas, and use registered taxis where possible."),
    },
    "tipping": {
        "examples": [
            "do you tip in mauritius",
            "is tipping expected in mauritius",
            "how much should i tip",
            "do i need to tip taxi drivers",
            "is a service charge included at restaurants",
        ],
        "response": ("Tipping isn't obligatory, but it's appreciated. Many hotels "
                     "and restaurants add a service charge already; if not, "
                     "around 10% for good service, and small tips for drivers, "
                     "porters, or tour guides, are common."),
    },
    "connectivity": {
        "examples": [
            "where can i buy a sim card in mauritius",
            "is there good wifi in mauritius",
            "does my phone work in mauritius",
            "how do i get internet access on the island",
            "can i use mobile data in mauritius",
            "is there roaming coverage in mauritius",
        ],
        "response": ("Local prepaid SIM cards are available cheaply at the "
                     "airport and in town from the main network operators, and "
                     "most hotels, resorts, and cafes offer free Wi-Fi. Check "
                     "with your home carrier about international roaming rates "
                     "before you rely on that instead."),
    },
    "budget": {
        "examples": [
            "how much does a week in mauritius cost",
            "is mauritius an expensive destination",
            "what's a typical daily budget for mauritius",
            "is mauritius cheap or expensive",
            "how much money should i bring",
        ],
        "response": ("Costs vary widely: budget guesthouses and street food can "
                     "make for an affordable trip, while all-inclusive beach "
                     "resorts and imported goods push costs up quickly. "
                     "Mid-range travellers typically budget more than they would "
                     "for similar trips in mainland Africa or South Asia."),
    },
    "electricity": {
        "examples": [
            "what plug type is used in mauritius",
            "what voltage does mauritius use",
            "do i need an adapter for mauritius",
            "what socket type is used there",
        ],
        "response": ("Mauritius uses UK-style Type G three-pin sockets at 230V, "
                     "50Hz. Visitors from countries with different plug types or "
                     "voltages will need a travel adapter (and a converter for "
                     "devices not rated for 230V)."),
    },
    "health": {
        "examples": [
            "do i need vaccinations to travel to mauritius",
            "is malaria a risk in mauritius",
            "are there health risks in mauritius",
            "do i need travel insurance for mauritius",
            "what medical facilities are available there",
        ],
        "response": ("No special vaccinations are required for most travellers, "
                     "though routine vaccinations should be up to date, and a "
                     "yellow fever certificate may be required if arriving from "
                     "a country with risk of transmission. Mauritius is "
                     "malaria-free. Private clinics and hospitals are available "
                     "for tourists; travel insurance covering medical care is "
                     "recommended."),
    },
    "diving_snorkeling": {
        "examples": [
            "where's the best place to go scuba diving in mauritius",
            "is snorkeling good in mauritius",
            "best dive sites in mauritius",
            "can beginners try diving in mauritius",
        ],
        "response": ("Mauritius has good diving and snorkeling around its "
                     "lagoons and reefs, with popular spots near Flic en Flac, "
                     "Blue Bay Marine Park, and the north coast. Dive centres "
                     "offer trips and courses for both beginners and "
                     "experienced divers."),
    },
    "honeymoon": {
        "examples": [
            "is mauritius good for honeymoons",
            "is mauritius romantic for couples",
            "best honeymoon resorts in mauritius",
            "is mauritius a good place for a romantic getaway",
        ],
        "response": ("Mauritius is a popular honeymoon destination, with many "
                     "resorts offering couples' packages, private beach "
                     "dinners, and adults-only sections. Areas like Belle Mare, "
                     "Trou aux Biches, and the south-west coast are especially "
                     "popular for honeymooners."),
    },
    "family_kids": {
        "examples": [
            "what activities are good for kids in mauritius",
            "is mauritius family friendly",
            "things to do with children in mauritius",
            "are there family resorts in mauritius",
        ],
        "response": ("Mauritius is very family friendly: many resorts have kids' "
                     "clubs, calm lagoons for swimming, and activities like "
                     "glass-bottom boat trips, mini water parks, and the Ile aux "
                     "Cerfs day trip that work well for children."),
    },
    "flights": {
        "examples": [
            "how long is the flight to mauritius",
            "which airlines fly to mauritius",
            "is there a direct flight to mauritius",
            "how far is mauritius by plane",
        ],
        "response": ("Flight time depends heavily on where you're departing "
                     "from, ranging from a few hours from parts of Africa and "
                     "Asia to 11+ hours from Europe. Air Mauritius and several "
                     "major international carriers operate routes to Sir "
                     "Seewoosagur Ramgoolam International Airport (MRU)."),
    },
    "airport_transfer": {
        "examples": [
            "how far is the airport from the main hotels",
            "how do i get from the airport to my hotel",
            "is airport transfer included with resorts",
            "how long is the drive from the airport",
        ],
        "response": ("The airport is in the south-east of the island; drive "
                     "times to popular resort areas range from about 20 minutes "
                     "(south coast) to over an hour (north or west coast). Most "
                     "hotels can arrange transfers, and taxis are readily "
                     "available at the airport."),
    },
    "regions": {
        "examples": [
            "which part of mauritius should i stay in",
            "best area to stay in mauritius",
            "north or south mauritius for a holiday",
            "which coast is best in mauritius",
        ],
        "response": ("The north (Grand Baie, Trou aux Biches) is lively with "
                     "nightlife and shopping, the west (Flic en Flac) is known "
                     "for sunsets and diving, the east (Belle Mare) is quieter "
                     "with long beaches, and the south is wilder and less "
                     "developed, good for nature and surfing."),
    },
    "nightlife": {
        "examples": [
            "what's the nightlife like in mauritius",
            "are there bars and clubs in mauritius",
            "where can i go out at night in mauritius",
        ],
        "response": ("Nightlife is concentrated mainly around Grand Baie in the "
                     "north, with bars, clubs, and casinos. Elsewhere on the "
                     "island, evenings tend to be quieter, centered around hotel "
                     "bars and restaurants."),
    },
    "shopping": {
        "examples": [
            "where can i go shopping in mauritius",
            "best markets in mauritius",
            "what souvenirs should i buy in mauritius",
        ],
        "response": ("Port Louis's Central Market and Caudan Waterfront are "
                     "popular for souvenirs, textiles, and local crafts. There "
                     "are also modern shopping malls like Bagatelle and Grand "
                     "Baie La Croisette for a more conventional shopping trip."),
    },
    "driving_license": {
        "examples": [
            "can i drive with a foreign driving license in mauritius",
            "do i need an international driving permit for mauritius",
            "what's needed to rent a car in mauritius",
        ],
        "response": ("Visitors can generally drive using a valid foreign "
                     "driving license for a limited period (commonly around a "
                     "year), though some rental companies ask for an "
                     "International Driving Permit as well. Mauritius drives on "
                     "the left."),
    },
    "day_trips": {
        "examples": [
            "can i visit rodrigues island as a day trip",
            "day trips from mauritius",
            "can i visit other islands near mauritius",
        ],
        "response": ("Rodrigues is a separate island about 90 minutes away by "
                     "plane and is usually visited as a multi-day trip rather "
                     "than a day trip. Closer options for a day out include Ile "
                     "aux Cerfs, Ile aux Benitiers, and catamaran cruises along "
                     "the coast."),
    },
    "lgbtq": {
        "examples": [
            "is mauritius lgbtq friendly",
            "is it safe to travel to mauritius as a gay couple",
            "lgbtq travel mauritius",
        ],
        "response": ("Attitudes in Mauritius are generally conservative, and "
                     "public displays of affection are uncommon for any couple. "
                     "Tourist resorts tend to be discreet and welcoming, but "
                     "travellers should be aware local social norms are more "
                     "conservative than in many Western countries."),
    },
    "solo_female": {
        "examples": [
            "is mauritius safe for solo female travelers",
            "can a woman travel alone in mauritius",
            "solo female travel mauritius safety",
        ],
        "response": ("Many women travel solo in Mauritius without major "
                     "incident, and tourist areas are generally safe. As "
                     "anywhere, it's sensible to avoid isolated areas at night, "
                     "use licensed taxis, and keep an eye on drinks in bars."),
    },
    "weddings": {
        "examples": [
            "can foreigners get married in mauritius",
            "how do i organize a wedding in mauritius",
            "is mauritius good for destination weddings",
        ],
        "response": ("Mauritius is a popular destination wedding location, and "
                     "foreigners can marry there, though it involves submitting "
                     "documents in advance and a waiting/residency period before "
                     "the ceremony. Most resorts and wedding planners handle the "
                     "paperwork for you."),
    },
    "duty_free": {
        "examples": [
            "what's the duty free allowance in mauritius",
            "how much alcohol can i bring into mauritius",
            "customs allowance for mauritius",
        ],
        "response": ("Mauritius allows a personal duty-free allowance for "
                     "items like alcohol, tobacco, and perfume within set "
                     "limits, and requires declaration of amounts above those "
                     "limits or of restricted goods. Check current customs "
                     "limits before you travel, since allowances are updated "
                     "periodically."),
    },
    "tap_water": {
        "examples": [
            "can you drink tap water in mauritius",
            "is tap water safe in mauritius",
            "should i drink bottled water in mauritius",
        ],
        "response": ("Tap water in most tourist areas and hotels is treated and "
                     "generally considered safe, though many visitors and "
                     "locals prefer bottled water, especially outside main "
                     "towns, as a precaution."),
    },
    "insects": {
        "examples": [
            "are there mosquitoes in mauritius",
            "is dengue a risk in mauritius",
            "should i bring mosquito repellent to mauritius",
        ],
        "response": ("Mosquitoes are present, especially in the wetter months, "
                     "and there have occasionally been localized dengue cases. "
                     "Repellent and covering up in the evenings are sensible "
                     "precautions, though this isn't a major deterrent for most "
                     "visitors."),
    },
    "public_holidays": {
        "examples": [
            "what are the public holidays in mauritius",
            "is mauritius closed on public holidays",
            "when are the holidays in mauritius",
        ],
        "response": ("Mauritius celebrates a mix of public holidays reflecting "
                     "its multicultural population, including New Year, "
                     "Independence Day (12 March), Diwali, Eid, Christmas, and "
                     "others. Some shops and services may have reduced hours on "
                     "these dates."),
    },
    "entry_requirements": {
        "examples": [
            "are there any covid entry requirements for mauritius",
            "what are the current entry rules for mauritius",
            "do i need any health documents to enter mauritius",
        ],
        "response": ("Entry health requirements change over time and by "
                     "traveller origin, so check the latest guidance from the "
                     "official Mauritian government or your airline shortly "
                     "before departure rather than relying on older "
                     "information."),
    },
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
        "response": ("Mauritius was uninhabited until Arab and Portuguese sailors "
                     "passed through, then settled by the Dutch in 1638 (who named "
                     "it after Prince Maurice of Orange), abandoned in 1710, and "
                     "colonized by the French from 1715 as Isle de France. Britain "
                     "took control in 1810 and the island reverted to the name "
                     "Mauritius. It gained independence on 12 March 1968 and "
                     "became a republic within the Commonwealth in 1992."),
    },
    "population": {
        "examples": [
            "what is the population of mauritius",
            "how many people live in mauritius",
            "how big is the population there",
            "population of mauritius",
            "how many people live on the island",
        ],
        "response": ("Mauritius has a population of roughly 1.27 million people "
                     "(2026 estimate), making it one of the more densely "
                     "populated countries in Africa. The population is "
                     "multi-ethnic, with Indo-Mauritian, Creole, Sino-Mauritian, "
                     "and Franco-Mauritian communities."),
    },
    "geography": {
        "examples": [
            "where is mauritius located",
            "how big is mauritius",
            "what continent is mauritius part of",
            "is mauritius near madagascar",
            "geography of mauritius",
            "how far is mauritius from africa",
        ],
        "response": ("Mauritius is a volcanic island in the Indian Ocean, about "
                     "2,000 km off the south-east coast of Africa and roughly "
                     "900 km east of Madagascar. It covers about 2,040 square "
                     "kilometers and is part of the Mascarene Islands, along "
                     "with Reunion and Rodrigues (the latter being part of the "
                     "Republic of Mauritius)."),
    },
    "government": {
        "examples": [
            "what type of government does mauritius have",
            "is mauritius a democracy",
            "who governs mauritius",
            "is mauritius a republic",
            "what is the political system in mauritius",
        ],
        "response": ("Mauritius is a parliamentary republic and a member of the "
                     "Commonwealth, generally regarded as one of Africa's most "
                     "stable democracies. It has a President as head of state "
                     "and a Prime Minister who leads the government, with a "
                     "National Assembly elected by voters."),
    },
    "economy": {
        "examples": [
            "what is the economy of mauritius based on",
            "what industries does mauritius have",
            "how does mauritius make money",
            "is mauritius a rich country",
            "what is mauritius known for economically",
        ],
        "response": ("Mauritius has a diversified upper-middle to high-income "
                     "economy built on tourism, textiles and garment "
                     "manufacturing, sugar production, financial and offshore "
                     "services, and a growing information and communication "
                     "technology sector."),
    },
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
        "response": ("Cyclone activity changes every season, so treat this as "
                     "a snapshot rather than live information: as of mid-2026, "
                     "Mauritius hadn't taken a direct major hit that season, "
                     "though Intense Tropical Cyclone Dudzai passed about "
                     "300 km southeast of the island in mid-January 2026 and "
                     "triggered a warning for dangerous swells and coastal "
                     "flooding risk, and a tropical depression brought "
                     "flooding to Rodrigues in March 2026. For the current "
                     "cyclone status and official warnings, check the "
                     "Mauritius Meteorological Services site "
                     "(metservice.intnet.mu) directly, especially if you're "
                     "traveling during the cyclone season (November to May, "
                     "peaking January to March)."),
    },
}

# Words that appear in almost every example regardless of topic and would
# otherwise dominate the similarity score (this was the main bug found in
# the evaluation: "mauritius" alone caused near-random, over-confident
# matches). These are removed on top of the standard English stop words.
DOMAIN_STOPWORDS = {
    "mauritius", "mauritian", "mauritians", "island", "here", "there",
    "tell", "know", "want", "explain",
}
STOPWORDS = list(sk_text.ENGLISH_STOP_WORDS.union(DOMAIN_STOPWORDS))

FALLBACK = ("I'm not fully sure about that one. I can help with topics like "
            "visas, safety, budget, transport, accommodation, food, weather, "
            "connectivity, and more - try rephrasing, or ask something else "
            "about visiting Mauritius.")

# A match is only trusted if its score clears MIN_SCORE *and* beats the
# runner-up intent by at least MIN_MARGIN - a single strong, unambiguous hit,
# not just "the least-bad of 34 options".
MIN_SCORE = 0.22
MIN_MARGIN = 0.04


class TourismChatbot:
    def __init__(self, intents=INTENTS):
        self.intents = intents
        self.utterances, self.labels = [], []
        for name, data in intents.items():
            for ex in data["examples"]:
                self.utterances.append(ex)
                self.labels.append(name)

        self._stopwords = set(STOPWORDS)
        self.vectorizer = TfidfVectorizer(analyzer=self._analyze)
        self.matrix = self.vectorizer.fit_transform(self.utterances)
        self.vocab = set(self.vectorizer.vocabulary_.keys())
        # only stemmed unigrams (no spaces) are worth spell-correcting token by token
        self._stem_vocab = sorted(w for w in self.vocab if " " not in w)

    def _analyze(self, doc):
        """Tokenize -> drop stopwords -> stem -> emit unigrams AND bigrams
        of the *surviving* stemmed tokens. Doing stopword removal and
        stemming ourselves (rather than relying on TfidfVectorizer's
        built-ins) keeps the vocabulary small, meaningful, and stable so
        typo-correction and stemming can both target it."""
        tokens = [t for t in _tokenize(doc) if t not in self._stopwords]
        stems = [simple_stem(t) for t in tokens]
        bigrams = [f"{a} {b}" for a, b in zip(stems, stems[1:])]
        return stems + bigrams

    def _correct_typos(self, query):
        """Best-effort fuzzy correction of each word against the fitted
        vocabulary, so small typos ('atractions', 'curency', 'accomodation')
        don't silently vanish as out-of-vocabulary words. Stopwords and very
        short words are skipped - they're either discarded anyway or too
        short for fuzzy matching to be reliable (e.g. 'at' ~ 'eat')."""
        words = _tokenize(query)
        corrected = []
        for w in words:
            if w in self._stopwords or len(w) < 4:
                corrected.append(w)
                continue
            stem = simple_stem(w)
            if stem in self.vocab:
                corrected.append(w)
                continue
            match = difflib.get_close_matches(stem, self._stem_vocab, n=1, cutoff=0.90)
            if match and abs(len(match[0]) - len(stem)) <= 2:
                corrected.append(match[0])
            else:
                corrected.append(w)
        return " ".join(corrected)

    def respond_debug(self, query):
        cleaned = self._correct_typos(query)
        sims = cosine_similarity(self.vectorizer.transform([cleaned]), self.matrix)[0]

        # aggregate to one score per intent (max over that intent's examples)
        best_per_label = {}
        for label, score in zip(self.labels, sims):
            if score > best_per_label.get(label, -1):
                best_per_label[label] = score
        ranked = sorted(best_per_label.items(), key=lambda kv: kv[1], reverse=True)

        top_label, top_score = ranked[0]
        runner_up_score = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = top_score - runner_up_score
        is_confident = (top_score >= MIN_SCORE) and (margin >= MIN_MARGIN)

        return {
            "matched_label": top_label if is_confident else None,
            "score": float(top_score),
            "margin": float(margin),
            "is_confident": is_confident,
            "response": self.intents[top_label]["response"] if is_confident else None,
            "ranked": ranked[:3],
            "corrected_query": cleaned,
        }

    def respond(self, query):
        """Kept for backwards compatibility / offline-only use: returns the
        static FALLBACK text when not confident, instead of deferring to an
        LLM."""
        dbg = self.respond_debug(query)
        return dbg["response"] if dbg["is_confident"] else FALLBACK


if __name__ == "__main__":
    bot = TourismChatbot()
    print("Mauritius Chatbot (v2, offline layer only). Type 'quit' to exit.\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in {"quit", "exit"}:
            break
        print("Bot:", bot.respond(q), "\n")
