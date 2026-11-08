# This Python file contains various 'tool' functions that are used by the agent to perform different steps
# in the process of recommending movies to a user based on their requests and the movie database.

# Library imports
import ast
import os
import pandas as pd


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
    """Rank a list of candidate movies by their average rating and return the top results.
    
    Args:
        movie_ids: A list of integer movie IDs (e.g., [2448, 2571, 2600]) found by previous tools.
        top_n: The maximum number of movies to return (default is 5).
        
    Returns:
        A sorted list of the top N movie dictionaries.
    """
    if not movie_ids:
        return []
    
    # Python does the heavy lifting: look up the exact IDs in the dataframe
    candidates = _catalog_df[_catalog_df['movieId'].isin(movie_ids)]
    
    # Sort them deterministically by rating
    ranked = candidates.sort_values(by='mean', ascending=False).head(top_n)
    
    return ranked.to_dict('records')


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