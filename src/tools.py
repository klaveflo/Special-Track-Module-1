"""Tool functions the agent can call.

Convention: tools RAISE ValueError with a helpful message on bad input.
The agent loop catches it and feeds the message back to the model as the observation.
"""

import ast

from src.catalog import (
    catalog,
    normalize_genre,
    to_int,
    to_records,
)
from src.memory import get_user_preferences

MAX_RESULTS = 10          # hard cap on rows returned to the model (keeps context small)
MIN_RATINGS = 3           # catalog entries with fewer ratings are never recommended
LIKE_BONUS = 0.30         # score added per genre shared with a liked movie
DISLIKE_PENALTY = 0.40    # score subtracted per genre shared with a disliked movie
MAX_ADJUSTMENT = 1.0      # preference adjustment is clipped to +/- this value

BASE_COLUMNS = ["movieId", "title", "year", "genres", "mean", "count", "score"]


def _as_list(value) -> list:
    """Accept a list, a single value, a comma-separated string or a stringified list.

    Small models send things like "['Romance']", "[]" or "Romance, Horror" for list parameters.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                parsed = text.strip("[]")
            if isinstance(parsed, (list, tuple, set)):
                return list(parsed)
            text = str(parsed)
        return [part.strip() for part in text.split(",") if part.strip()]
    return [value]


def _blank_to_none(value):
    """Small models fill unused optional parameters with junk like '', 'null', '[]' or 0."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in ("", "none", "null", "n/a", "0", "[]"):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value == 0:
        return None
    return value


def search_movies(query: str, limit: int = 5) -> list[dict]:
    """Search the catalog for movies whose title contains the query text.
    Use this to find the movieId of a specific title the user mentions.

    Args:
        query: Part of a movie title (e.g., 'Matrix', 'Star Wars').
        limit: Maximum number of results (default 5, maximum 10).

    Returns:
        Matching movies, most-rated first, each with movieId, title, year, genres, mean, count, score.
    """
    query = str(query).strip()
    if not query:
        raise ValueError("query must not be empty.")
    limit = min(max(to_int(limit, "limit"), 1), MAX_RESULTS)

    mask = catalog["title"].str.contains(query, case=False, na=False, regex=False)
    matches = catalog[mask].sort_values("count", ascending=False).head(limit)
    return to_records(matches, BASE_COLUMNS)


def filter_movies(
    genre: str = None,
    year: int = None,
    year_from: int = None,
    year_to: int = None,
    exclude_genres: list[str] = None,
    limit: int = 10,
) -> list[dict]:
    """Find candidate movies by genre and/or release year, best-scoring first.
    Use this when the user asks for a category or time period rather than a specific title.
    Only pass the parameters you need and leave the others out.

    Args:
        genre: Genre the movie must have (e.g., 'Action', 'Sci-Fi'). Case-insensitive.
        year: Exact release year (e.g., 1999). Ignored if year_from or year_to is given.
        year_from: Earliest release year, inclusive (e.g., 1990 for 'the 90s').
        year_to: Latest release year, inclusive (e.g., 1999 for 'the 90s').
        exclude_genres: Genres the movie must NOT have (e.g., ['Horror']).
        limit: Maximum number of results (default 10, maximum 10).

    Returns:
        Up to `limit` movies sorted by a rating-count-adjusted score, each with
        movieId, title, year, genres, mean, count, score. An empty list means no match.
    """
    limit = min(max(to_int(limit, "limit"), 1), MAX_RESULTS)
    genre, year, year_from, year_to = map(_blank_to_none, (genre, year, year_from, year_to))
    df = catalog[catalog["count"] >= MIN_RATINGS]

    if year_from is not None or year_to is not None:
        year = None  # a range wins over an exact year (models often send both)

    if year is not None:
        df = df[df["year"] == to_int(year, "year")]
    if year_from is not None:
        df = df[df["year"] >= to_int(year_from, "year_from")]
    if year_to is not None:
        df = df[df["year"] <= to_int(year_to, "year_to")]

    if genre is not None:
        wanted = normalize_genre(genre)
        df = df.loc[df["genres"].map(lambda gs: wanted in gs).astype(bool)]

    for g in _as_list(exclude_genres):
        if _blank_to_none(g) is None:
            continue
        banned = normalize_genre(g)
        df = df.loc[~df["genres"].map(lambda gs: banned in gs).astype(bool)]

    df = df.sort_values(["score", "count"], ascending=False).head(limit)
    return to_records(df, BASE_COLUMNS)


def _clean_ids(movie_ids) -> list[int]:
    """Accept ints, numeric strings, or {'movieId': ...} dicts; silently skip junk."""
    ids = []
    for item in _as_list(movie_ids):
        if isinstance(item, dict) and "movieId" in item:
            item = item["movieId"]
        try:
            ids.append(to_int(item))
        except ValueError:
            continue
    return ids


def rank_movies(movie_ids: list[int], prefs: dict | None = None, top_n: int = 5) -> list[dict]:
    """Plain Python ranking function (NOT an LLM tool).

    Pass prefs=None to ignore memory. A stage-gated loop can call this directly
    with preferences it loaded itself.
    """
    df = catalog[catalog["movieId"].isin(_clean_ids(movie_ids))]
    prefs = prefs or {"likes": [], "dislikes": []}

    likes = prefs.get("likes", [])
    dislikes = prefs.get("dislikes", [])
    seen_ids = {m["movieId"] for m in likes} | {m["movieId"] for m in dislikes}
    liked_genres = {g for m in likes for g in m.get("genres", [])}
    disliked_genres = {g for m in dislikes for g in m.get("genres", [])}

    df = df[~df["movieId"].isin(seen_ids)]

    ranked = []
    for rec in to_records(df, BASE_COLUMNS):
        genres = set(rec["genres"])
        got = sorted(genres & liked_genres)
        lost = sorted(genres & disliked_genres)
        bonus = LIKE_BONUS * len(got)
        penalty = DISLIKE_PENALTY * len(lost)
        adjustment = max(-MAX_ADJUSTMENT, min(MAX_ADJUSTMENT, bonus - penalty))

        notes = []
        if got:
            notes.append(f"+{bonus:.2f} liked genres ({', '.join(got)})")
        if lost:
            notes.append(f"-{penalty:.2f} disliked genres ({', '.join(lost)})")

        rec["personalized_score"] = round(rec["score"] + adjustment, 3)
        rec["preference_adjustment"] = round(adjustment, 3)
        rec["preference_note"] = "; ".join(notes) or "No stored genre adjustments applied"
        ranked.append(rec)

    ranked.sort(key=lambda r: (r["personalized_score"], r["count"]), reverse=True)
    return ranked[:top_n]


def rank_candidates(movie_ids: list[int], top_n: int = 5, use_memory: bool = True) -> list[dict]:
    """Rank candidate movies by adjusted rating score, optionally personalized with stored preferences.
    Call this after filter_movies or search_movies to pick the best candidates.

    Scoring rule:
        1. Base = rating-count-adjusted score (few ratings are pulled toward the global average).
        2. +0.30 per genre shared with a liked movie, -0.40 per genre shared with a disliked movie
           (total adjustment capped at +/-1.0).
        3. Movies the user already liked or disliked are excluded.
        Set use_memory to false to ignore stored preferences for this request.

    Args:
        movie_ids: Integer movie IDs taken from filter_movies/search_movies results.
        top_n: Maximum number of movies to return (default 5).
        use_memory: Apply stored likes/dislikes (default true).

    Returns:
        Up to top_n movies with personalized_score, preference_adjustment and preference_note.
    """
    top_n = max(to_int(top_n, "top_n"), 1)
    prefs = get_user_preferences() if use_memory else None
    return rank_movies(movie_ids, prefs, top_n)


def get_movie_details(movie_id: int) -> dict:
    """Get the catalog record for one movie by its movieId.
    Returns only data the catalog has (title, year, genres, rating stats). Runtime and plot are
    not available and are always reported as 'Unknown'.

    Args:
        movie_id: The integer movieId exactly as returned by search_movies or filter_movies.

    Returns:
        A dictionary with movieId, title, year, genres, mean, count, score, runtime, plot.
    """
    movie_id = to_int(movie_id, "movie_id")
    row = catalog[catalog["movieId"] == movie_id]
    if row.empty:
        raise ValueError(f"No movie found with ID {movie_id}.")

    details = to_records(row, BASE_COLUMNS)[0]
    details.setdefault("runtime", "Unknown")  # setdefault: real data later will not be overwritten
    details.setdefault("plot", "Unknown")
    return details
