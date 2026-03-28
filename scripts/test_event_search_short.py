# scripts/test_event_search_short.py
import os, requests
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

TOKEN = os.getenv("EVENTBRITE_TOKEN")
headers = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}
params = {
    "location.address": "London",
    "start_date.range_start": "2025-06-01T00:00:00Z",
    "start_date.range_end":   "2025-06-07T00:00:00Z",
    "expand": "venue,category",
    "page": 1
}
r = requests.get("https://www.eventbriteapi.com/v3/events/search/", headers=headers, params=params, timeout=15)
print("status:", r.status_code)
print("url:", r.url)
print("response (first 2000 chars):")
print(r.text[:2000])
