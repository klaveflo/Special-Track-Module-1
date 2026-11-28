"""Bounded multi-step movie recommendation agent (Week 9).

Python owns the order of the stages and the final answer. The model only decides:
  1. how to interpret the request (structured output)
  2. which search tools to call and with which arguments
Memory, ranking, detail lookup and the answer text are plain Python, so the agent
cannot skip them or invent facts. Every action is recorded in a trace.
"""

import json
import time
from typing import Literal

from ollama import chat
from pydantic import BaseModel, ValidationError

from src.catalog import normalize_genre
from src.memory import get_user_preferences, save_feedback
from src.retrieval import search_descriptions
from src.tools import filter_movies, get_movie_details, rank_candidates, search_movies

MODEL = "llama3.1:8b"
OPTIONS = {"temperature": 0.0, "num_predict": 256}   # hard cap on tokens per model call
MAX_SEARCH_STEPS = 3   # model turns in the search stage (a new turn only starts if nothing was found)
MAX_CALLS_PER_TURN = 2  # extra tool calls in one model message are ignored
SEARCH_TOOLS = {"filter_movies": filter_movies, "search_descriptions": search_descriptions}

INTERPRET_PROMPT = (
    "You extract a movie request as JSON. Extract only what the user explicitly says. Do not guess. "
    "intent: 'feedback' if the user says they liked or disliked a specific movie, otherwise 'recommend'. "
    "genre: one movie genre. year_from and year_to: for a decade use its first and last year, "
    "for a single year use it twice. mood: themes or style such as 'twist ending' or 'atmospheric'. "
    "exclusions: things the user does not want. maximum_runtime: minutes. "
    "ignore_history: true only if the user asks to ignore their past preferences. "
    "title, sentiment, reason: only for feedback."
)

SEARCH_PROMPT = (
    "You find candidate movies for the request. Use filter_movies for genre, years and excluded genres, "
    "and search_descriptions (query = the mood text) for themes or styles. You may call both. "
    "Only pass the parameters you need. If a call returns nothing, relax one constraint and try again."
)


class MovieRequest(BaseModel):
    intent: Literal["recommend", "feedback"]
    genre: str | None = None
    year_from: int | None = None
    year_to: int | None = None
    mood: str | None = None
    maximum_runtime: int | None = None
    exclusions: list[str] = []
    ignore_history: bool = False
    title: str | None = None            # feedback only
    sentiment: Literal["like", "dislike"] | None = None
    reason: str | None = None


def log(trace, stage, actor, tool, args, summary, status="ok"):
    trace.append({"stage": stage, "actor": actor, "tool": tool, "args": args,
                  "summary": summary, "status": status})


def print_trace(trace):
    for i, e in enumerate(trace, 1):
        print(f"{i:>2}. {e['stage']:<9} {e['actor']:<7} {e['tool']}({e['args']}) "
              f"-> {e['status']}: {e['summary']}")


def format_movies(movies):
    return "\n".join(
        f"{m['movieId']} | {m['title']} ({m['year']}) | {', '.join(m['genres'])} "
        f"| mean {m['mean']} (n={m['count']})" for m in movies)


def prepare(req):
    """Map the request onto tool arguments. Returns (args, notes, problem)."""
    notes, mood, genre = [], req.mood, None
    if req.genre:
        try:
            genre = normalize_genre(req.genre)
        except ValueError:
            mood = f"{mood or ''} {req.genre}".strip()
            notes.append(f"'{req.genre}' is not a catalog genre, so it was used as search text.")
    exclude = []
    for item in req.exclusions:
        try:
            exclude.append(normalize_genre(item))
        except ValueError:
            notes.append(f"'{item}' cannot be excluded: the catalog only knows genres.")
    if req.maximum_runtime:
        notes.append("The catalog has no runtimes, so the maximum runtime was not applied.")

    problem = None
    if genre and genre in exclude:
        problem = f"'{genre}' is both requested and excluded."
    elif req.year_from and req.year_to and req.year_from > req.year_to:
        problem = "the start year is later than the end year."
    args = {"genre": genre, "year_from": req.year_from, "year_to": req.year_to,
            "exclude_genres": exclude or None, "mood": mood}
    return args, notes, problem


def default_search(args):
    """What Python searches for if the model makes no usable call."""
    filters = {k: v for k, v in args.items() if v and k != "mood"}
    if filters or not args["mood"]:
        return "filter_movies", filters          # no filters at all: top-rated movies
    return "search_descriptions", {"query": args["mood"]}


def run_search_tool(name, call_args, found, evidence):
    results = SEARCH_TOOLS[name](**call_args)
    for m in results:
        found[m["movieId"]] = m
        if m.get("matched_tags"):
            evidence[m["movieId"]] = m["matched_tags"]
    return results


def collect_candidates(args, trace, chat_fn):
    """Search stage: the model picks the tools, Python executes and guards them."""
    request_text = "Find candidate movies for: " + ", ".join(
        f"{k}={v}" for k, v in args.items() if v)
    messages = [{"role": "system", "content": SEARCH_PROMPT},
                {"role": "user", "content": request_text}]
    found, evidence, seen = {}, {}, set()

    for _ in range(MAX_SEARCH_STEPS):
        msg = chat_fn(model=MODEL, messages=messages, tools=list(SEARCH_TOOLS.values()),
                      options=OPTIONS).message
        messages.append(msg)
        if not msg.tool_calls:
            log(trace, "search", "model", "(stop)", {}, "no further tool call")
            break
        for call in msg.tool_calls[:MAX_CALLS_PER_TURN]:
            name, call_args = call.function.name, dict(call.function.arguments)
            key = (name, json.dumps(call_args, sort_keys=True, default=str))
            status = "ok"
            if name not in SEARCH_TOOLS:
                obs, status = f"Tool '{name}' is not available.", "blocked"
            elif key in seen:
                obs, status = "Repeated call blocked. Change the arguments or stop.", "blocked"
            else:
                seen.add(key)
                try:
                    results = run_search_tool(name, call_args, found, evidence)
                    obs = (format_movies(results) + "\nCandidates stored.") if results \
                        else "No results. Relax one constraint and try again."
                except (ValueError, TypeError) as e:     # bad values or unknown argument names
                    obs, status = f"Error: {e}", "error"
            log(trace, "search", "model", name, call_args, obs.splitlines()[0][:80], status)
            messages.append({"role": "tool", "tool_name": name, "content": obs})
        if found:
            break                                        # retry only while nothing was found

    if not found and not seen:                           # the model never made a usable call
        name, call_args = default_search(args)
        results = run_search_tool(name, call_args, found, evidence)
        log(trace, "search", "python", name, call_args, f"fallback: {len(results)} movies")
    return found, evidence


def handle_feedback(req, trace):
    if not (req.title and req.sentiment):
        return "Which movie, and did you like or dislike it?"
    matches = search_movies(req.title)
    exact = [m for m in matches if m["title"].lower() == req.title.strip().lower()]
    chosen = exact or matches
    log(trace, "feedback", "python", "search_movies", {"query": req.title}, f"{len(chosen)} match(es)")
    if not chosen:
        return f"I could not find '{req.title}' in the catalog."
    if len(chosen) > 1:
        options = "; ".join(f"{m['title']} ({m['year']})" for m in chosen)
        return f"Which one do you mean? {options}"
    movie = chosen[0]
    saved = save_feedback(movie["movieId"], req.sentiment, req.reason)
    log(trace, "feedback", "python", "save_feedback",
        {"movie_id": movie["movieId"], "sentiment": req.sentiment}, saved["status"])
    return f"Saved: {saved['sentiment']} for {movie['title']} ({movie['year']})."


def build_answer(picks, details, evidence, notes):
    lines = []
    for label, rec, det in zip(("Primary", "Alternative"), picks, details):
        lines.append(f"{label}: {det['title']} ({det['year']}), ID {det['movieId']}")
        lines.append(f"  Genres: {', '.join(det['genres'])} | mean {det['mean']} from {det['count']} ratings")
        if evidence.get(det["movieId"]):
            lines.append(f"  Matching tags: {', '.join(evidence[det['movieId']])}")
        if rec["preference_adjustment"]:
            lines.append(f"  Memory: {rec['preference_note']}")
    lines += [f"Note: {n}" for n in notes]
    return "\n".join(lines)


def run_agent(user_prompt, chat_fn=chat):
    """Stages: interpret -> (feedback | search -> memory -> rank -> details -> answer)."""
    start, trace = time.perf_counter(), []

    def result(answer, movie_ids=()):
        return {"answer": answer, "movie_ids": list(movie_ids), "trace": trace,
                "seconds": round(time.perf_counter() - start, 1)}

    try:
        response = chat_fn(model=MODEL, format=MovieRequest.model_json_schema(), options=OPTIONS,
                           messages=[{"role": "system", "content": INTERPRET_PROMPT},
                                     {"role": "user", "content": user_prompt}])
        req = MovieRequest.model_validate_json(response.message.content)
    except ValidationError:
        return result("I could not understand that request. Please rephrase it.")
    log(trace, "interpret", "model", "interpret", {}, req.model_dump_json(exclude_defaults=True))

    if req.intent == "feedback":
        return result(handle_feedback(req, trace))

    args, notes, problem = prepare(req)
    if problem:
        return result(f"Your request contradicts itself: {problem} Please revise it.")

    found, evidence = collect_candidates(args, trace, chat_fn)
    if not found:
        wanted = ", ".join(f"{k}={v}" for k, v in args.items() if v) or "no constraints"
        return result(f"No movies found for: {wanted}. Please relax or change the request.")

    use_memory = not req.ignore_history
    if use_memory:
        prefs = get_user_preferences()
        log(trace, "memory", "python", "get_user_preferences", {},
            f"{len(prefs['likes'])} likes, {len(prefs['dislikes'])} dislikes")
    else:
        notes.append("Stored preferences were ignored as requested.")
    ranked = rank_candidates(list(found), top_n=5, use_memory=use_memory)
    log(trace, "rank", "python", "rank_candidates", {"n": len(found), "use_memory": use_memory},
        f"{len(ranked)} ranked")
    if not ranked:
        return result("All matching movies are already rated by you. Please try a different request.")

    picks = ranked[:2]
    details = [get_movie_details(m["movieId"]) for m in picks]
    log(trace, "details", "python", "get_movie_details", {"ids": [d["movieId"] for d in details]},
        "primary and alternative")
    return result(build_answer(picks, details, evidence, notes), [d["movieId"] for d in details])