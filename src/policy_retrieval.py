
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


CHUNK_FILE = "outputs/policy_chunks.json"

chunks = []
vectorizer = None
tfidf_matrix = None


# Build TF-IDF index
def build_index(policy_chunks):

    global chunks, vectorizer, tfidf_matrix

    chunks = policy_chunks

    documents = [chunk["text"] for chunk in chunks]

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(documents)


# Load original policy chunks
with open(CHUNK_FILE, "r", encoding="utf-8") as file:
    initial_chunks = json.load(file)

build_index(initial_chunks)


# Retrieve top relevant chunks
def retrieve(query, k=3):

    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    top_indices = similarities.argsort()[::-1][:k]

    results = []

    for index in top_indices:

        results.append({
            "source": chunks[index]["source"],
            "score": round(float(similarities[index]), 4),
            "text": chunks[index]["text"]
        })

    return results
