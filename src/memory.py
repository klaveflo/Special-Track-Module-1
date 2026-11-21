import json
import os
import tempfile
from datetime import datetime, timezone

from src.catalog import DATA_DIR, catalog, to_int

PREFS_PATH = DATA_DIR / "user_preferences.json"


def _default_preferences() -> dict:
    """Clean, minimal preference schema with no personal identifiers."""
    return {"user_id": "test_user_01", "likes": [], "dislikes": []}


def _write_preferences(prefs: dict) -> None:
    """Atomic write: a crash mid-write can never leave a half-written file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=DATA_DIR, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(prefs, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, PREFS_PATH)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def get_user_preferences() -> dict:
    """Retrieve the test user's stored movie likes and dislikes (with reasons).

    Returns:
        A dictionary containing 'user_id', 'likes', and 'dislikes'.
    """
    if not PREFS_PATH.exists():
        return _default_preferences()  # no side effect: nothing is written on read

    try:
        with PREFS_PATH.open("r", encoding="utf-8") as f:
            prefs = json.load(f)
        if not isinstance(prefs, dict):
            raise json.JSONDecodeError("not an object", "", 0)
    except json.JSONDecodeError:
        # Keep the broken file for inspection instead of silently erasing the history
        os.replace(PREFS_PATH, PREFS_PATH.with_suffix(".json.bak"))
        return _default_preferences()

    # Tolerate manually edited files
    prefs.setdefault("user_id", "test_user_01")
    prefs.setdefault("likes", [])
    prefs.setdefault("dislikes", [])
    return prefs


def save_feedback(movie_id: int, sentiment: str, reason: str = None) -> dict:
    """Record explicit user feedback ('like' or 'dislike') for a specific movie ID.
    Use this tool whenever the user says they liked, loved, disliked, or hated a movie.

    Args:
        movie_id: The unique integer ID of the movie exactly as returned by search_movies.
        sentiment: Either 'like' or 'dislike'.
        reason: Optional short explanation for why the user liked or disliked the movie.

    Returns:
        A confirmation dictionary with the saved entry.
    """
    clean_id = to_int(movie_id, "movie_id")

    s = str(sentiment).strip().lower()
    if s in ("like", "liked", "love", "loved"):
        bucket, label = "likes", "like"
    elif s in ("dislike", "disliked", "hate", "hated"):
        bucket, label = "dislikes", "dislike"
    else:
        raise ValueError(f"Invalid sentiment {sentiment!r}. Must be 'like' or 'dislike'.")

    match = catalog[catalog["movieId"] == clean_id]
    if match.empty:
        raise ValueError(f"Movie ID {clean_id} does not exist in the catalog.")

    row = match.iloc[0]
    entry = {
        "movieId": clean_id,
        "title": str(row["title"]),
        "genres": list(row["genres"]),
        "reason": str(reason).strip() if reason else None,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    # A movie can only be in one bucket, so a new rating overrides the old one
    prefs = get_user_preferences()
    prefs["likes"] = [m for m in prefs["likes"] if m["movieId"] != clean_id]
    prefs["dislikes"] = [m for m in prefs["dislikes"] if m["movieId"] != clean_id]
    prefs[bucket].append(entry)

    _write_preferences(prefs)
    return {"status": "saved", "sentiment": label, "entry": entry}


def reset_preferences() -> dict:
    """Reset stored preferences to an empty baseline. NOT exposed to the LLM."""
    prefs = _default_preferences()
    _write_preferences(prefs)
    return prefs
