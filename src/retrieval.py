"""Tag-based retrieval over the MovieLens catalog (TF-IDF + cosine similarity).

The text corpus is built from MovieLens user tags plus title and genres, for the
tagged movies only. Importing this module is cheap: the corpus and the index are
built on the first call of search_descriptions(), so a missing tags.csv only
breaks this one tool.
"""

from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.catalog import DATA_DIR, MIN_RATINGS, catalog

RAW_TAGS_PATH = DATA_DIR / "ml-latest-small" / "tags.csv"   # original MovieLens download
CORPUS_PATH = DATA_DIR / "movie_text_corpus.csv"             # optional documentation export
TAG_STOPLIST = {"in netflix queue"}   # one user's watchlist habit, not a description of the movie
STOP_WORDS = list(ENGLISH_STOP_WORDS | {"movie", "movies", "film", "films"})
MAX_RESULTS = 5


def normalize_text(text: str) -> str:
    """Lowercase and drop hyphens so 'Sci-Fi' and 'sci-fi' both become 'scifi'."""
    return str(text).lower().strip().replace("-", "")


def build_corpus() -> pd.DataFrame:
    """One row per tagged catalog movie with its unique cleaned tags and a 'text' document."""
    if not RAW_TAGS_PATH.exists():
        raise FileNotFoundError(
            f"MovieLens tags file not found at {RAW_TAGS_PATH}. Download MovieLens "
            "latest-small from GroupLens and unzip it into the data/ folder."
        )

    raw = pd.read_csv(RAW_TAGS_PATH)
    raw["tag"] = raw["tag"].str.lower().str.strip()   # keep original spelling for display
    raw = raw[(raw["tag"] != "") & ~raw["tag"].isin(TAG_STOPLIST)]

    # SET of unique tags per movie, so one heavily tagged movie cannot dominate by repetition
    movie_tags = (
        raw.groupby("movieId")["tag"]
        .agg(lambda s: sorted(set(s)))
        .rename("tags")
        .reset_index()
    )

    corpus = (
        catalog[catalog["count"] >= MIN_RATINGS]
        .merge(movie_tags, on="movieId", how="inner")
        .reset_index(drop=True)
    )
    corpus["text"] = corpus.apply(
        lambda r: " ".join(
            [normalize_text(r["title"])]
            + [normalize_text(g) for g in r["genres"]]
            + [normalize_text(t) for t in r["tags"]]
        ),
        axis=1,
    )
    return corpus


@lru_cache(maxsize=1)
def _get_index():
    """Build the corpus and TF-IDF index once, on first use."""
    corpus = build_corpus()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words=STOP_WORDS, sublinear_tf=True)
    matrix = vectorizer.fit_transform(corpus["text"])
    return corpus, vectorizer, matrix


def search_descriptions(query: str) -> list[dict]:
    """Find movies whose tags, genres or title match a free-text request (TF-IDF + cosine similarity).
    Use this for moods, themes or styles that genre and year cannot express, such as 'twist ending'
    or 'atmospheric'. Only movies that have user-written tags can be found. Tags are short subjective
    keywords from a few MovieLens users, not plot descriptions.

    Args:
        query: Free-text request, e.g. 'surreal dark comedy'.

    Returns:
        Up to 5 movies with similarity above 0, best first. Each has movieId, title, year, genres,
        mean, count, similarity, matched_tags and matched_genres (the evidence for the match).
        An empty list means nothing in the tagged corpus matches.
    """
    query = str(query).strip()
    if not query:
        raise ValueError("query must not be empty.")

    corpus, vectorizer, matrix = _get_index()
    clean_query = normalize_text(query)
    sims = cosine_similarity(vectorizer.transform([clean_query]), matrix).ravel()
    order = np.lexsort((-corpus["score"].to_numpy(), -sims))   # similarity first, rating breaks ties
    q_tokens = set(clean_query.split())

    results = []
    for i in order[:MAX_RESULTS]:
        if sims[i] <= 0:
            break                                              # never pad with non-matches
        row = corpus.iloc[i]
        results.append({
            "movieId": int(row["movieId"]),
            "title": row["title"],
            "year": int(row["year"]) if pd.notna(row["year"]) else None,
            "genres": list(row["genres"]),
            "mean": round(float(row["mean"]), 3),
            "count": int(row["count"]),
            "similarity": round(float(sims[i]), 3),
            "matched_tags": [t for t in row["tags"] if q_tokens & set(normalize_text(t).split())],
            "matched_genres": [g for g in row["genres"] if q_tokens & set(normalize_text(g).split())],
        })
    return results
