import ast
import json
from pathlib import Path
import pandas as pd

# Resolve paths relative to this module so imports work from any working directory
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CATALOG_PATH = DATA_DIR / "cleaned_movie_data.csv"
PREFS_PATH = DATA_DIR / "user_preferences.json"

# Load catalog once to validate movie IDs and attach genres to stored feedback
_catalog_df = pd.read_csv(CATALOG_PATH)
_catalog_df["genres"] = _catalog_df["genres"].apply(ast.literal_eval)


def _default_preferences() -> dict:
    """Return a clean, minimal preference schema with no personal identifiers."""
    return {
        "user_id": "test_user_01",
        "likes": [],
        "dislikes": [],
    }


def _write_preferences(prefs: dict) -> None:
    """Persist the preferences dictionary to disk as formatted UTF-8 JSON."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with PREFS_PATH.open("w", encoding="utf-8") as f:
        json.dump(prefs, f, indent=2, ensure_ascii=False)


def get_user_preferences() -> dict:
    """Retrieve the test user's stored movie likes, dislikes, and feedback reasons.
    Use this tool before recommending movies to check if the user has existing preferences.

    Returns:
        A dictionary containing 'user_id', 'likes', and 'dislikes'.
    """
    try:
        with PREFS_PATH.open("r", encoding="utf-8") as f:
            prefs = json.load(f)
        # Ensure required keys exist even if the file was manually edited
        prefs.setdefault("user_id", "test_user_01")
        prefs.setdefault("likes", [])
        prefs.setdefault("dislikes", [])
        return prefs
    except (FileNotFoundError, json.JSONDecodeError):
        prefs = _default_preferences()
        _write_preferences(prefs)
        return prefs


def save_feedback(movie_id: int, sentiment: str, reason: str = None) -> dict:
    """Record explicit user feedback ('like' or 'dislike') for a specific movie ID.
    Use this tool whenever the user says they liked, loved, disliked, or hated a movie.

    Args:
        movie_id: The unique integer ID of the movie from the catalog (e.g., 2571).
        sentiment: String, either 'like' or 'dislike' indicating the user's sentiment.
        reason: Optional short explanation for why the user liked or disliked the movie.

    Returns:
        A confirmation dictionary with the saved entry, or an error dictionary if invalid.
    """
    # Defensively cast movie_id (handles LLMs passing strings like "2571")
    try:
        clean_id = int(movie_id)
    except (ValueError, TypeError):
        return {"error": f"Invalid movie_id '{movie_id}'. Must be an integer."}

    # Normalize and validate sentiment
    clean_sentiment = str(sentiment).strip().lower()
    if clean_sentiment in ("like", "liked", "love", "loved"):
        target_bucket = "likes"
        normalized_label = "like"
    elif clean_sentiment in ("dislike", "disliked", "hate", "hated"):
        target_bucket = "dislikes"
        normalized_label = "dislike"
    else:
        return {"error": f"Invalid sentiment '{sentiment}'. Must be 'like' or 'dislike'."}

    # Verify movie exists in the local catalog and fetch its metadata
    match = _catalog_df[_catalog_df["movieId"] == clean_id]
    if match.empty:
        return {"error": f"Movie ID {clean_id} does not exist in the catalog."}

    movie_row = match.iloc[0]
    entry = {
        "movieId": clean_id,
        "title": str(movie_row["title"]),
        "genres": list(movie_row["genres"]),
        "reason": str(reason).strip() if reason else None,
    }

    # Load existing preferences, deduplicate this movie ID, and save
    prefs = get_user_preferences()
    prefs["likes"] = [m for m in prefs["likes"] if m["movieId"] != clean_id]
    prefs["dislikes"] = [m for m in prefs["dislikes"] if m["movieId"] != clean_id]
    prefs[target_bucket].append(entry)

    _write_preferences(prefs)
    return {
        "status": "saved",
        "sentiment": normalized_label,
        "entry": entry,
    }


def reset_preferences() -> dict:
    """Reset the stored user preferences back to an empty baseline state.

    Returns:
        The freshly reset preferences dictionary.
    """
    prefs = _default_preferences()
    _write_preferences(prefs)
    return prefs