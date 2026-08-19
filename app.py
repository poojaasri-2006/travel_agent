"""
Travel Planner Agent — FastAPI + LangGraph deployment
Deploy target: Render (runs `uvicorn app:app`)
 
Env vars required (set in Render's dashboard, NOT in code):
    GOOGLE_API_KEY   -> your Gemini API key from https://aistudio.google.com/apikey
"""
 
import os
import json
import random
import re
from datetime import datetime, timedelta
 
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langserve import add_routes
 
# ---------------------------------------------------------------------------
# 1. API key
# ---------------------------------------------------------------------------
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY environment variable is not set. "
        "Add it under Render > your service > Environment."
    )
 
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3,
    max_retries=6,
)
 
 
# ---------------------------------------------------------------------------
# 2. Tools
# ---------------------------------------------------------------------------
@tool
def weather_check(city: str, date: str) -> dict:
    """Get the weather forecast for a city on a specific date (format YYYY-MM-DD).
    Returns max/min temperature (C) and chance of rain."""
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
 
        return {
            "city": city,
            "date": date,
            "temp_max_c": daily["temperature_2m_max"][0],
            "temp_min_c": daily["temperature_2m_min"][0],
            "rain_chance_pct": daily["precipitation_probability_max"][0],
        }
 
    except requests.exceptions.RequestException as e:
        return {"error": f"Network error while checking weather: {e}"}
    except (KeyError, IndexError, ValueError) as e:
        return {"error": f"Unexpected response format from weather API: {e}"}
 
 
@tool
def flight_price_search(origin: str, destination: str, depart_date: str, return_date: str) -> dict:
    """Estimate round-trip flight price between two cities for given dates.
    Returns an estimated price in USD. This is a MOCK — replace with a real
    flight API (Amadeus/Skyscanner) for production use."""
    seed = sum(ord(c) for c in (origin + destination + depart_date))
    random.seed(seed)
    base_price = random.randint(250, 900)
 
    return {
        "origin": origin,
        "destination": destination,
        "depart_date": depart_date,
        "return_date": return_date,
        "estimated_price_usd": base_price,
        "note": "Mock estimate — connect a real flight API for live pricing.",
    }
 
 
@tool
def attraction_finder(city: str, max_results: int = 6) -> list:
    """Find popular tourist attractions in a city using Wikipedia search.
    Returns a list of attraction names with short descriptions."""
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
            return [{"error": f"No attractions found for '{city}'"}]
 
        return results
 
    except requests.exceptions.RequestException as e:
        return [{"error": f"Network error while finding attractions: {e}"}]
    except (KeyError, ValueError) as e:
        return [{"error": f"Unexpected response format from Wikipedia API: {e}"}]
 
 
@tool
def itinerary_builder(destination: str, start_date: str, num_days: int,
                       attractions: list, budget_usd: float) -> dict:
    """Build a day-by-day itinerary by distributing attractions across the trip days.
    attractions can be a list of dicts with 'name' and 'description', OR a list
    of plain strings (attraction names). Returns a structured itinerary."""
 
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
 
    return {
        "title": f"{num_days}-Day Trip to {destination}",
        "budget_usd": budget_usd,
        "days": days,
    }
 
 
tools = [weather_check, flight_price_search, attraction_finder, itinerary_builder]
 
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
    tools=tools,
    prompt=system_prompt,
)
 
 
# ---------------------------------------------------------------------------
# 4. Request schema + helpers
# ---------------------------------------------------------------------------
class TripRequest(BaseModel):
    destination: str = Field(..., description="City/country the traveler wants to visit")
    origin: str = Field(..., description="City the traveler is departing from")
    start_date: str = Field(..., description="Trip start date, YYYY-MM-DD")
    return_date: str = Field(..., description="Trip return date, YYYY-MM-DD")
    budget_usd: float = Field(..., description="Total trip budget in USD")
 
 
def extract_final_text(agent_result: dict) -> str:
    for msg in reversed(agent_result.get("messages", [])):
        if msg.__class__.__name__ != "AIMessage":
            continue
        content = getattr(msg, "content", "")
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text" and block.get("text", "").strip():
                    return block["text"]
    return ""
 
 
def extract_itinerary_json(agent_result: dict):
    for m in agent_result.get("messages", []):
        if getattr(m, "name", None) == "itinerary_builder":
            content = m.content
            try:
                return json.loads(content) if isinstance(content, str) else content
            except (json.JSONDecodeError, TypeError):
                return None
    return None
 
 
def run_trip_planner(payload: TripRequest) -> dict:
    start = datetime.strptime(payload.start_date, "%Y-%m-%d")
    end = datetime.strptime(payload.return_date, "%Y-%m-%d")
    num_days = max(1, (end - start).days)
 
    user_request = (
        f"Plan a {num_days}-day trip to {payload.destination}, departing from {payload.origin}, "
        f"starting {payload.start_date} and returning {payload.return_date}. "
        f"My total budget is ${payload.budget_usd}."
    )
 
    result = agent.invoke({"messages": [{"role": "user", "content": user_request}]})
 
    return {
        "destination": payload.destination,
        "origin": payload.origin,
        "start_date": payload.start_date,
        "return_date": payload.return_date,
        "budget_usd": payload.budget_usd,
        "summary": extract_final_text(result),
        "itinerary": extract_itinerary_json(result),
    }
 
 
# ---------------------------------------------------------------------------
# 5. FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Travel Planner Agent",
    version="1.0",
    description="Agent that builds a day-by-day travel itinerary given destination, dates, and budget.",
)
 
 
@app.post("/plan-trip")
async def plan_trip(req: TripRequest):
    try:
        return run_trip_planner(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trip planning failed: {e}")
 
 
# Optional: keep LangServe routes too (adds /travel-planner/invoke, /stream, /playground/)
add_routes(app, agent, path="/travel-planner")
 
 
# ---------------------------------------------------------------------------
# 6. Homepage
# ---------------------------------------------------------------------------
HOMEPAGE_HTML = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Travel Planner Agent</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/9.1.2/marked.min.js"></script>
  <style>
    body { font-family: system-ui, sans-serif; background:#f5f5f5; margin:0; padding:40px 16px; }
    .card { max-width: 640px; margin: 0 auto; background:#fff; border-radius:12px; padding:32px; box-shadow:0 1px 4px rgba(0,0,0,0.08); }
    h1 { font-size: 1.4rem; margin: 0 0 4px 0; }
    .subtitle { color:#666; margin-bottom:24px; }
    label { display:block; margin-top:16px; font-weight:600; font-size:0.9rem; }
    input[type=text], input[type=number], input[type=date] {
      width:100%; padding:10px; margin-top:6px; box-sizing:border-box;
      border:1px solid #ccc; border-radius:8px; font-size:0.95rem;
    }
    .row { display:flex; gap:16px; }
    .row > div { flex:1; }
    button {
      margin-top:24px; padding:12px 20px; width:100%; border:none; border-radius:8px;
      background:#3b5bfd; color:white; font-size:1rem; font-weight:600; cursor:pointer;
    }
    button:disabled { background:#a5a5a5; cursor:not-allowed; }
    #status { margin-top:16px; font-size:0.9rem; color:#555; }
    #resultBox { display:none; margin-top:24px; }
    #summaryOut {
      background:#fafafa; border:1px solid #eee; border-radius:8px;
      padding:18px 20px; font-size:0.92rem; line-height:1.55;
    }
    #summaryOut h1, #summaryOut h2, #summaryOut h3 { margin-top:1.2em; margin-bottom:0.4em; }
    #summaryOut ul { padding-left:1.2em; }
  </style>
</head>
<body>
  <div class="card">
    <h1>&#9992;&#65039; Travel Planner Agent</h1>
    <p class="subtitle">Enter your trip details and get a full day-by-day itinerary.</p>
 
    <form id="tripForm">
      <label for="destination">Destination</label>
      <input type="text" id="destination" placeholder="Tokyo, Japan" required />
 
      <label for="origin">Departing from</label>
      <input type="text" id="origin" placeholder="Hyderabad, India" required />
 
      <div class="row">
        <div>
          <label for="start_date">Start date</label>
          <input type="date" id="start_date" required />
        </div>
        <div>
          <label for="return_date">Return date</label>
          <input type="date" id="return_date" required />
        </div>
      </div>
 
      <label for="budget_usd">Budget (USD)</label>
      <input type="number" id="budget_usd" placeholder="1500" required />
 
      <button type="submit" id="submitBtn">Plan My Trip</button>
    </form>
 
    <div id="status"></div>
 
    <div id="resultBox">
      <div id="summaryOut"></div>
    </div>
  </div>
 
  <script>
    const form = document.getElementById("tripForm");
    const statusEl = document.getElementById("status");
    const resultBox = document.getElementById("resultBox");
    const summaryOut = document.getElementById("summaryOut");
    const submitBtn = document.getElementById("submitBtn");
 
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      resultBox.style.display = "none";
      submitBtn.disabled = true;
      statusEl.textContent = "Planning your trip... this can take 20-60 seconds.";
 
      const payload = {
        destination: document.getElementById("destination").value,
        origin: document.getElementById("origin").value,
        start_date: document.getElementById("start_date").value,
        return_date: document.getElementById("return_date").value,
        budget_usd: parseFloat(document.getElementById("budget_usd").value),
      };
 
      try {
        const res = await fetch("/plan-trip", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        const contentType = res.headers.get("content-type") || "";
        if (!contentType.includes("application/json")) {
          const text = await res.text();
          statusEl.textContent = "Server error (status " + res.status + "): " + text.slice(0, 200);
          return;
        }
        const data = await res.json();
        if (!res.ok) {
          statusEl.textContent = "Error: " + (data.detail || res.statusText);
        } else {
          statusEl.textContent = "Done.";
          resultBox.style.display = "block";
          summaryOut.innerHTML = marked.parse(data.summary || "(no summary returned)");
        }
      } catch (err) {
        statusEl.textContent = "Request failed: " + err;
      } finally {
        submitBtn.disabled = false;
      }
    });
  </script>
</body>
</html>
"""
 
 
@app.get("/", response_class=HTMLResponse)
async def homepage():
    return HOMEPAGE_HTML
 
 
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
