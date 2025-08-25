#!/usr/bin/env python3

import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# -----------------------
# Helper Functions
# -----------------------

def normalize_url(url: str) -> str:
    """Ensure the website starts with https:// if missing."""
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url
    return url

def scrape_website(url: str, max_pages: int = 20):
    """Scrape up to `max_pages` pages from the site starting at `url`."""
    visited = set()
    to_visit = [url]
    pages = []

    while to_visit and len(pages) < max_pages:
        current_url = to_visit.pop(0)
        if current_url in visited:
            continue
        visited.add(current_url)

        try:
            response = requests.get(current_url, timeout=5)
            if "text/html" not in response.headers.get("Content-Type", ""):
                continue
            soup = BeautifulSoup(response.text, "html.parser")

            text = soup.get_text(separator=" ", strip=True)
            pages.append((current_url, text))

            for a_tag in soup.find_all("a", href=True):
                link = urljoin(current_url, a_tag["href"])
                if link.startswith(url):
                    to_visit.append(link)

        except Exception as e:
            print(f"Error scraping {current_url}: {e}")

    return pages

def rank_pages(pages, query_keywords):
    """Rank pages based on TF-IDF and cosine similarity with query keywords."""
    documents = [text for _, text in pages]
    urls = [url for url, _ in pages]

    # TF-IDF
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    doc_matrix = vectorizer.fit_transform(documents)

    # Query vector
    query = " ".join(query_keywords)
    query_vector = vectorizer.transform([query])

    # Debugging shape info
    print(f"Document matrix shape: {doc_matrix.shape}")
    print(f"Query vector shape: {query_vector.shape}")

    # Cosine similarity
    similarities = cosine_similarity(query_vector, doc_matrix).flatten()

    ranked_indices = np.argsort(similarities)[::-1]
    ranked_pages = [(urls[i], similarities[i]) for i in ranked_indices]

    return ranked_pages

# -----------------------
# Streamlit UI
# -----------------------

st.title("🔎 Website Page Importance & Relevance Ranker")

website = st.text_input("Enter Website URL (e.g., www.egain.coms or https://egain.com):")
keywords = st.text_input("Relevance Topic / Query (comma-separated keywords, optional):")

if st.button("Analyze Website"):
    if website:
        normalized_url = normalize_url(website)
        st.write(f"Fetching pages from: **{normalized_url}** ...")

        with st.spinner("Scraping website..."):
            pages = scrape_website(normalized_url, max_pages=15)

        if not pages:
            st.error("No pages could be scraped. Try a different site.")
        else:
            query_keywords = [kw.strip() for kw in keywords.split(",")] if keywords else []
            ranked_pages = rank_pages(pages, query_keywords)

            st.subheader("📊 Ranked Pages")
            for url, score in ranked_pages[:10]:
                st.write(f"{url} — **Relevance Score: {score:.4f}**")

