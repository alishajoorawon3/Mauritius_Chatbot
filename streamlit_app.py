"""
Mauritius Tourism Chatbot - self-contained Streamlit app.
Deployable on Streamlit Community Cloud (needs only this file + requirements.txt).
"""

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------- intents
INTENTS = {
    "best_time_to_visit": {
        "examples": ["when is the best time to visit mauritius",
                     "best season to travel to mauritius",
                     "when should i go to mauritius"],
        "response": ("Mauritius is warm year-round. The driest, sunniest months "
                     "are May to December; January to March is hotter and the "
                     "main cyclone season.")},
    "attractions": {
        "examples": ["what are the best attractions in mauritius",
                     "things to do in mauritius", "tourist spots in mauritius"],
        "response": ("Popular attractions include Le Morne Brabant, Chamarel Seven "
                     "Coloured Earths, Black River Gorges, Ile aux Cerfs, and "
                     "Port Louis.")},
    "transport": {
        "examples": ["how do i get around mauritius",
                     "is there public transport in mauritius",
                     "should i rent a car in mauritius"],
        "response": ("Getting around is usually by rental car, taxi, or the public "
                     "bus network, which covers most of the island.")},
    "accommodation": {
        "examples": ["where should i stay in mauritius",
                     "hotel recommendations mauritius", "villa or hotel in mauritius"],
        "response": ("Mauritius offers beach resort hotels, villas, apartments, and "
                     "guesthouses. Non-hotel options have grown in recent years.")},
    "currency": {
        "examples": ["what currency is used in mauritius",
                     "do i need to exchange money", "can i pay in euros in mauritius"],
        "response": ("The local currency is the Mauritian Rupee (MUR). Major hotels "
                     "often accept euros or US dollars, but carry some local "
                     "currency for smaller vendors.")},
    "weather": {
        "examples": ["what is the weather like in mauritius",
                     "is it rainy in mauritius", "mauritius temperature"],
        "response": ("Mauritius has a tropical climate: warm and dry May-December, "
                     "hotter and wetter January-March.")},
    "food": {
        "examples": ["what food should i try in mauritius",
                     "local cuisine mauritius", "what do people eat in mauritius"],
        "response": ("Mauritian cuisine blends Indian, Creole, Chinese, and French "
                     "influences - try dholl puri, rougaille, and biryani.")},
    "language": {
        "examples": ["what language is spoken in mauritius",
                     "do people speak english in mauritius"],
        "response": ("English is the official language and French is widely used; "
                     "most Mauritians also speak Creole.")},
}

FALLBACK = ("I'm not sure I understood that. I can help with: best time to visit, "
            "attractions, transport, accommodation, currency, weather, food, and "
            "language. Could you rephrase?")
THRESHOLD = 0.15


class TourismChatbot:
    def __init__(self, intents=INTENTS):
        self.intents = intents
        self.utterances, self.labels = [], []
        for name, data in intents.items():
            for ex in data["examples"]:
                self.utterances.append(ex)
                self.labels.append(name)
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.utterances)

    def respond(self, query):
        sims = cosine_similarity(self.vectorizer.transform([query.lower()]),
                                 self.matrix)[0]
        best = sims.argmax()
        if sims[best] < THRESHOLD:
            return FALLBACK
        return self.intents[self.labels[best]]["response"]


# ---------------------------------------------------------------- interface
st.set_page_config(page_title="Mauritius Tourism Chatbot", page_icon="🏝️")
st.title("🏝️ Mauritius Tourism Chatbot")
st.caption("Ask about the best time to visit, attractions, transport, "
           "accommodation, currency, weather, food, or language.")

if "bot" not in st.session_state:
    st.session_state.bot = TourismChatbot()
if "history" not in st.session_state:
    st.session_state.history = []

prompt = st.chat_input("Ask me about visiting Mauritius...")
if prompt:
    reply = st.session_state.bot.respond(prompt)
    st.session_state.history.append(("user", prompt))
    st.session_state.history.append(("assistant", reply))

for role, text in st.session_state.history:
    with st.chat_message(role):
        st.write(text)

with st.sidebar:
    st.subheader("Try asking about:")
    for name in INTENTS:
        st.markdown(f"- {name.replace('_', ' ').title()}")
    st.divider()
    st.caption("Prototype built for a BSc dissertation on tourism analytics "
               "and conversational AI for Mauritius.")
