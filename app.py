"""
Travel Planner Agent — FastAPI + LangServe deployment
Deploy target: Render (or any host that runs `uvicorn app:app`)
 
Env vars required (set these in Render's dashboard, NOT in code):
    GOOGLE_APIKEY   -> your Gemini API key from https://aistudio.google.com/apikey
"""
 
import os
import json
import random
import re
from datetime import datetime, timedelta
 
import requests
from fastapi import FastAPI
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langserve import add_routes
 
# ---------------------------------------------------------------------------
# 1. API key — read from environment (Render injects this from your dashboard
#    settings, it should NEVER be hardcoded in this file)
# ---------------------------------------------------------------------------
# reads your Render env var named GOOGLE_APIKEY
api_key = os.environ.get("GOOGLE_APIKEY")
if not api_key:
    raise RuntimeError(
        "GOOGLE_APIKEY environment variable is not set. "
        "Add it under Render > your service > Environment."
    )
 
# but LangChain's Gemini wrapper specifically looks for GOOGLE_API_KEY
os.environ["GOOGLE_API_KEY"] = api_key
 
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    temperature=0.3,
    max_retries=6,       # auto-retry on transient 503/overload errors
    timeout=60,          # fail fast instead of hanging indefinitely
)
 
 
# ---------------------------------------------------------------------------
# 2. Tools
# ---------------------------------------------------------------------------
@tool
def weather_check(city: str, date: str) -> dict:
    """Get the weather forecast for a city on a specific date (format YYYY-MM-DD).
    Returns max/min temperature (C) and chance of rain."""
    print(f"[weather_check] called: city={city}, date={date}", flush=True)
    try:
        geo_resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
            timeout=10,
        )
        if geo_resp.status_code != 200:
            return {"error": f"Geocoding API returned status {geo_resp.status_code} for '{city}'"}
 
        geo = geo_resp.json()
        if not geo.get("results"):
            return {"error": f"Could not find location: {city}"}
 
        lat = geo["results"][0]["latitude"]
        lon = geo["results"][0]["longitude"]
 
        forecast_resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "timezone": "auto",
                "start_date": date,
                "end_date": date,
            },
            timeout=10,
        )
        if forecast_resp.status_code != 200:
            return {"error": f"Forecast API returned status {forecast_resp.status_code}"}
 
        forecast = forecast_resp.json()
        daily = forecast.get("daily", {})
        if not daily.get("time"):
            return {"city": city, "date": date, "note": "Forecast unavailable this far out, assume seasonal average."}
 
        result = {
            "city": city,
            "date": date,
            "temp_max_c": daily["temperature_2m_max"][0],
            "temp_min_c": daily["temperature_2m_min"][0],
            "rain_chance_pct": daily["precipitation_probability_max"][0],
        }
        print(f"[weather_check] done: {result}", flush=True)
        return result
 
    except requests.exceptions.RequestException as e:
        print(f"[weather_check] network error: {e}", flush=True)
        return {"error": f"Network error while checking weather: {e}"}
    except (KeyError, IndexError, ValueError) as e:
        print(f"[weather_check] format error: {e}", flush=True)
        return {"error": f"Unexpected response format from weather API: {e}"}
 
 
@tool
def flight_price_search(origin: str, destination: str, depart_date: str, return_date: str) -> dict:
    """Estimate round-trip flight price between two cities for given dates.
    Returns an estimated price in USD. This is a MOCK — replace with a real
    flight API (Amadeus/Skyscanner) for production use."""
    print(f"[flight_price_search] called: {origin} -> {destination}", flush=True)
    seed = sum(ord(c) for c in (origin + destination + depart_date))
    random.seed(seed)
    base_price = random.randint(250, 900)
 
    result = {
        "origin": origin,
        "destination": destination,
        "depart_date": depart_date,
        "return_date": return_date,
        "estimated_price_usd": base_price,
        "note": "Mock estimate — connect a real flight API for live pricing.",
    }
    print(f"[flight_price_search] done: {result}", flush=True)
    return result
 
 
@tool
def attraction_finder(city: str, max_results: int = 6) -> list:
    """Find popular tourist attractions in a city using Wikipedia search.
    Returns a list of attraction names with short descriptions."""
    print(f"[attraction_finder] called: city={city}", flush=True)
    try:
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": f"tourist attractions in {city}",
                "format": "json",
                "srlimit": max_results,
            },
            timeout=10,
        )
        if resp.status_code != 200:
            return [{"error": f"Wikipedia API returned status {resp.status_code}"}]
 
        search = resp.json()
        results = []
        for item in search.get("query", {}).get("search", []):
            snippet = re.sub("<.*?>", "", item.get("snippet", ""))
            results.append({"name": item["title"], "description": snippet})
 
        if not results:
            print(f"[attraction_finder] no results for {city}", flush=True)
            return [{"error": f"No attractions found for '{city}'"}]
 
        print(f"[attraction_finder] done: {len(results)} results", flush=True)
        return results
 
    except requests.exceptions.RequestException as e:
        print(f"[attraction_finder] network error: {e}", flush=True)
        return [{"error": f"Network error while finding attractions: {e}"}]
    except (KeyError, ValueError) as e:
        print(f"[attraction_finder] format error: {e}", flush=True)
        return [{"error": f"Unexpected response format from Wikipedia API: {e}"}]
 
 
@tool
def itinerary_builder(destination: str, start_date: str, num_days: int,
                       attractions: list, budget_usd: float) -> dict:
    """Build a day-by-day itinerary by distributing attractions across the trip days.
    attractions can be a list of dicts with 'name' and 'description', OR a list
    of plain strings (attraction names). Returns a structured itinerary."""
    print(f"[itinerary_builder] called: destination={destination}, num_days={num_days}", flush=True)
 
    if isinstance(attractions, str):
        try:
            attractions = json.loads(attractions)
        except json.JSONDecodeError:
            attractions = [attractions]
 
    normalized = []
    for a in attractions:
        if isinstance(a, dict):
            normalized.append({
                "name": a.get("name", "Unnamed attraction"),
                "description": a.get("description", ""),
            })
        elif isinstance(a, str):
            normalized.append({"name": a, "description": ""})
 
    if not normalized:
        normalized = [{"name": f"Explore {destination}", "description": ""}]
 
    start = datetime.strptime(start_date, "%Y-%m-%d")
    days = []
 
    per_day = max(1, len(normalized) // max(1, num_days))
    idx = 0
 
    for d in range(num_days):
        date = (start + timedelta(days=d)).strftime("%Y-%m-%d")
        day_stops = normalized[idx: idx + per_day] or normalized[:1]
        idx += per_day
 
        stops = [
            {
                "name": a["name"],
                "time": "Morning" if i == 0 else "Afternoon",
                "blurb": a["description"][:120] if a["description"] else "",
            }
            for i, a in enumerate(day_stops)
        ]
        days.append({"day_label": f"Day {d+1} ({date})", "stops": stops})
 
    result = {
        "title": f"{num_days}-Day Trip to {destination}",
        "budget_usd": budget_usd,
        "days": days,
    }
    print(f"[itinerary_builder] done: {len(days)} days built", flush=True)
    return result
 
 
# ---------------------------------------------------------------------------
# 3. Agent
# ---------------------------------------------------------------------------
system_prompt = """You are a Travel Planner Agent.
Given a destination, travel dates, and budget, you must:
1. Check the weather for the destination on the trip dates.
2. Search for flight price estimates.
3. Find real attractions in the destination.
4. Use itinerary_builder to assemble a day-by-day plan.
Always call itinerary_builder LAST, after gathering the other data.
Present the final itinerary clearly, including weather and flight cost context."""
 
agent = create_react_agent(
    model=llm,
    tools=[weather_check, flight_price_search, attraction_finder, itinerary_builder],
    prompt=system_prompt,
)
 
 
# ---------------------------------------------------------------------------
# 4. FastAPI app + LangServe route
#    Once deployed, this exposes:
#      /travel-planner/invoke      -> POST, single request/response
#      /travel-planner/stream      -> POST, streamed response
#      /travel-planner/playground/ -> interactive browser UI
#      /docs                       -> auto-generated API docs
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Travel Planner Agent",
    version="1.0",
    description="Agent that builds a day-by-day travel itinerary given destination, dates, and budget.",
)
 
add_routes(app, agent, path="/travel-planner")
 
print("[startup] Travel Planner Agent initialized and routes mounted.", flush=True)
 
 
@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Travel Planner Agent is live.",
        "playground": "/travel-planner/playground/",
        "docs": "/docs",
    }
 
 
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
 
