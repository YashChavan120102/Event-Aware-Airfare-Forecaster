# scripts/test_eventbrite_user.py
import os, requests
# try to load .env if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

TOKEN = os.getenv("EVENTBRITE_TOKEN")
if not TOKEN:
    print("No EVENTBRITE_TOKEN found in .env")
    raise SystemExit(1)

headers = {"Authorization": f"Bearer {TOKEN}"}
r = requests.get("https://www.eventbriteapi.com/v3/users/me/", headers=headers, timeout=15)
print("status:", r.status_code)
print("url:", r.url)
print("response (truncated 1000 chars):")
print(r.text[:1000])
