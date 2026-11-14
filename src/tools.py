# This Python file contains various 'tool' functions that are used by the agent to perform different steps
# in the process of recommending movies to a user based on their requests and the movie database.

# Library imports
import ast
import os
import pandas as pd
from src.memory import get_user_preferences


# Robust path resolution to the data folder
current_dir = os.path.dirname(os.path.abspath(__file__))
catalog_path = os.path.join(current_dir, '../data/cleaned_movie_data.csv')

# Load movie catalog data into a pandas DataFrame
_catalog_df = pd.read_csv(catalog_path)

# Convert the 'genres' column from string representation of list to actual list
_catalog_df['genres'] = _catalog_df['genres'].apply(ast.literal_eval)


# search_movies function to find movies based on a search query
def search_movies(query: str) -> list[dict]:
    """Search the catalog for movies by their title or keywords.
    
    Args:
        query: The specific movie title or keyword to search for (e.g., 'Matrix', 'Star Wars').
        
    Returns:
        A list of movie dictionaries that contain the query in their title.
    """
    # Perform a case-insensitive search on the title column
    matches = _catalog_df[_catalog_df['title'].str.contains(query, case=False, na=False)]
    
    return matches.to_dict('records')


# filter_movies function to filter movies based on genre and/or release year
def filter_movies(genre: str = None, year: int = None) -> list[dict]:
    """Filter the movie catalog by specific genre and/or release year.
    Use this tool when the user asks for categories or time periods rather than specific titles.
    
    Args:
        genre: The movie genre to filter by (e.g., 'Action', 'Comedy', 'Sci-Fi').
        year: The specific release year to filter by (e.g., 1999, 2005).
        
    Returns:
        A list of movie dictionaries matching the requested criteria.
    """
    candidates = _catalog_df.copy()
    
    # Deterministically enforce the minimum rating threshold (baseline safety)
    candidates = candidates[candidates['count'] >= 3]
    
    if year is not None:
        try:
            # Defensively force input year to be an integer, rejecting non-integer formats
            clean_year = int(year)
            candidates = candidates[candidates['year'] == clean_year]
        except ValueError:
            # reject non-integer year inputs and return an error message
            return [{"error": f"Invalid year format: {year}. Must be an integer."}]
        
    if genre is not None:
        candidates = candidates[candidates['genres'].apply(lambda x: genre in x)]
        
    return candidates.to_dict('records')


# rank_candidated function to rank candidate movies based on their average rating
def rank_candidates(movie_ids: list[int], top_n: int = 5) -> list[dict]:
    """Rank candidate movies by combining catalog rating with stored user preferences.
    Always call this tool after searching or filtering to personalize the order of results.

    Scoring Rule:
        1. Exclude any movieId already present in user 'dislikes' or 'likes' (to recommend unseen titles).
        2. Start with the movie's catalog 'mean' rating.
        3. Add +0.30 for each genre matching a genre from the user's liked movies.
        4. Subtract -0.40 for each genre matching a genre from the user's disliked movies.

    Args:
        movie_ids: A list of integer movie IDs (e.g., [2571, 2600, 3175]) from search/filter results.
        top_n: Maximum number of ranked movies to return (default is 5).

    Returns:
        A sorted list of up to top_n movie dictionaries with 'personalized_score' and 'preference_note'.
    """
    if not movie_ids:
        return []

    # Defensively normalize IDs (handles strings or dicts if the LLM formats arguments loosely)
    clean_ids = []
    for item in movie_ids:
        if isinstance(item, dict) and "movieId" in item:
            clean_ids.append(int(item["movieId"]))
        else:
            try:
                clean_ids.append(int(item))
            except (ValueError, TypeError):
                continue

    if not clean_ids:
        return []

    candidates = _catalog_df[_catalog_df["movieId"].isin(clean_ids)].copy()
    if candidates.empty:
        return []

    # Load persistent user preferences
    prefs = get_user_preferences()
    liked_ids = {m["movieId"] for m in prefs.get("likes", [])}
    disliked_ids = {m["movieId"] for m in prefs.get("dislikes", [])}

    liked_genres = {g for m in prefs.get("likes", []) for g in m.get("genres", [])}
    disliked_genres = {g for m in prefs.get("dislikes", []) for g in m.get("genres", [])}

    # Rule 1: Exclude already-rated movies (both disliked and previously liked)
    excluded_ids = liked_ids | disliked_ids
    candidates = candidates[~candidates["movieId"].isin(excluded_ids)]
    if candidates.empty:
        return []

    # Rules 2-4: Compute personalized score per candidate
    scored_records = []
    for record in candidates.to_dict("records"):
        base_score = float(record.get("mean", 0.0))
        movie_genres = set(record.get("genres", []))

        matched_likes = sorted(movie_genres & liked_genres)
        matched_dislikes = sorted(movie_genres & disliked_genres)

        bonus = 0.30 * len(matched_likes)
        penalty = 0.40 * len(matched_dislikes)
        adjustment = round(bonus - penalty, 3)
        final_score = round(max(0.0, base_score + adjustment), 3)

        # Build a transparent explanation string the LLM can cite directly
        notes = []
        if matched_likes:
            notes.append(f"+{bonus:.2f} for liked genres ({', '.join(matched_likes)})")
        if matched_dislikes:
            notes.append(f"-{penalty:.2f} for disliked genres ({', '.join(matched_dislikes)})")

        record["personalized_score"] = final_score
        record["preference_adjustment"] = adjustment
        record["preference_note"] = (
            "; ".join(notes) if notes else "No stored genre adjustments applied"
        )
        scored_records.append(record)

    # Sort descending by personalized_score, using raw mean rating as tie-breaker
    scored_records.sort(
        key=lambda x: (x["personalized_score"], x.get("mean", 0.0)),
        reverse=True,
    )
    return scored_records[:top_n]


# get_movie_details function to retrieve detailed information about a specific movie
def get_movie_details(movie_id: int) -> dict:
    """Get the complete details for a specific movie using its unique movie ID.
    Use this tool when the user asks for more information, the plot, or runtime of a specific movie.
    
    Args:
        movie_id: The unique integer ID of the movie (e.g., 3578).
        
    Returns:
        A dictionary containing the movie's full details, or an error dictionary if not found.
    """
    # Find the specific movie by its ID
    movie_row = _catalog_df[_catalog_df['movieId'] == movie_id]
    
    if movie_row.empty:
        return {"error": f"No movie found with ID {movie_id}."}
        
    # Convert the single matching row to a dictionary
    movie_dict = movie_row.iloc[0].to_dict()
    
    # Explicitly enforce the "Unknown" fields requirement to prevent hallucination
    movie_dict['runtime'] = 'Unknown'
    movie_dict['plot'] = 'Unknown'
    
    return movie_dict