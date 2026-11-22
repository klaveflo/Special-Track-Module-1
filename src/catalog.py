"""Single place where the movie catalog is loaded, plus small shared helpers.

Both tools.py and memory.py import from here, so the CSV is read exactly once
and type handling is consistent everywhere.
"""

import ast
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CATALOG_PATH = DATA_DIR / "cleaned_movie_data.csv"

# How strongly a movie's score is pulled toward the global average.
# A movie needs roughly this many ratings before its own mean counts as much
# as the global mean. Tune it and document the value in your write-up.
PRIOR_STRENGTH = 10

# Catalog entries with fewer ratings are never recommended or retrieved.
MIN_RATINGS = 3


def _load_catalog() -> pd.DataFrame:
    df = pd.read_csv(CATALOG_PATH)
    df["genres"] = df["genres"].apply(ast.literal_eval)

    # Damped (Bayesian) mean: few ratings => score stays near the global mean.
    global_mean = (df["mean"] * df["count"]).sum() / df["count"].sum()
    df["score"] = (df["count"] * df["mean"] + PRIOR_STRENGTH * global_mean) / (
        df["count"] + PRIOR_STRENGTH
    )
    return df


catalog = _load_catalog()

VALID_GENRES = sorted({g for genres in catalog["genres"] for g in genres})
_GENRE_LOOKUP = {g.lower(): g for g in VALID_GENRES}


def to_int(value, name: str = "value") -> int:
    """Coerce LLM-supplied values like '1999', 1999.0 or 1999 to int, or raise a clear error."""
    try:
        return int(float(value))
    except (ValueError, TypeError):
        raise ValueError(f"{name} must be an integer, got {value!r}.")


def normalize_genre(genre: str) -> str:
    """Case-insensitive genre lookup. Raises an error that lists the valid genres."""
    key = str(genre).strip().lower()
    if key not in _GENRE_LOOKUP:
        raise ValueError(
            f"Unknown genre {genre!r}. Valid genres: {', '.join(VALID_GENRES)}."
        )
    return _GENRE_LOOKUP[key]


def to_records(df: pd.DataFrame, columns: list[str]) -> list[dict]:
    """DataFrame -> list of plain-Python dicts (JSON safe, rounded, year as int)."""
    records = []
    for row in df[columns].to_dict("records"):
        if "year" in row:
            row["year"] = int(row["year"]) if pd.notna(row["year"]) else None
        for key in ("mean", "score"):
            if key in row and pd.notna(row[key]):
                row[key] = round(float(row[key]), 3)
        if "count" in row:
            row["count"] = int(row["count"])
        if "movieId" in row:
            row["movieId"] = int(row["movieId"])
        records.append(row)
    return records
