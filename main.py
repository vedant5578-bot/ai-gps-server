"""AI GPS cloud server (FastAPI)."""
import io
import json
import math
import os
import subprocess
import time
from functools import lru_cache

import numpy as np
import requests
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from gtts import gTTS
from PIL import Image, ImageDraw

DEVICE_TOKEN = os.getenv("DEVICE_TOKEN", "change-me")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
STT_MODEL = os.getenv("STT_MODEL", "whisper-large-v3")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
TILE_URL = os.getenv("TILE_URL", "https://tile.openstreetmap.org/{z}/{x}/{y}.png")
OSRM_URL = os.getenv("OSRM_URL", "https://router.project-osrm.org")
UA = os.getenv("USER_AGENT", "esp32-ai-gps/1.0 (personal hobby project)")

W, H = 128, 160
app = FastAPI()
ROUTES: dict = {}
TTS_CACHE: dict = {}


def check(token: str):
    if token != DEVICE_TOKEN:
        raise HTTPException(status_code=401, detail="bad token")


# ----------------------------------------------------------------- geometry
def hav(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def nearest_idx(coords, lat, lon, start, end):
    k = math.cos(math.radians(lat))
    best, bi = 1e18, start
    for i in range(start, min(end, len(coords))):
        dy = coords[i][0] - lat
        dx = (coords[i][1] - lon) * k
        d = dx * dx + dy * dy
        if d < best:
            best, bi = d, i
    return bi, hav(lat, lon, coords[bi][0], coords[bi][1])


def latlon_to_px(lat, lon, z):
    s = 256 * 2 ** z
    x = (lon + 180.0) / 360.0 * s
    siny = min(max(math.sin(math.radians(lat)), -0.9999), 0.9999)
    y = (0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi)) * s
    return x, y


# ----------------------------------------------------------------- external services
def transcribe(wav: bytes) -> str:
    r = requests.post(
        "https://api.groq.com/openai/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        files={"file": ("audio.wav", wav, "audio/wav")},
        data={"model": STT_MODEL, "response_format": "json",
              "prompt": "Navigate to a place in India, for example India Gate Delhi, Chandigarh, Mohali, Amritsar."},
        timeout=40,
    )
    r.raise_for_status()
    return r.json().get("text", "").strip()


def extract_destination(text: str) -> str:
    if not GROQ_API_KEY:
        return text
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": LLM_MODEL,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content":
                        'The user speaks a navigation request (English/Hindi/Hinglish, may be noisy). '
                        'Reply ONLY with JSON: {"destination": "<place name suitable for a map search, in English, '
                        'include city if said>"}.'},
                    {"role": "user", "content": text},
                ],
            },
            timeout=25,
        )
        r.raise_for_status()
        data = json.loads(r.json()["choices"][0]["message"]["content"])
        return (data.get("destination") or text).strip()
    except Exception as e:
        print("extract_destination failed:", e)
        return text


def geocode(query: str, lat: float, lon: float):
    d = 1.5
    params = {"q": query, "format": "jsonv2", "limit": 1, "countrycodes": "in",
              "viewbox": f"{lon - d},{lat + d},{lon + d},{lat - d}"}
    r = requests.get("https://nominatim.openstreetmap.org/search", params=params,
                     headers={"User-Agent": UA}, timeout=15)
    r.raise_for_status()
    res = r.json()
    if not res:
        return None
    return float(res[0]["lat"]), float(res[0]["lon"]), res[0]["display_name"]


def instruction(m, road):
    t, mod = m.get("type", ""), m.get("modifier", "")
    on = f" onto {road}" if road else ""
    if t == "arrive":
        return "You have arrived at your destination"
    if t == "depart":
        return f"Start driving{on}"
    if t in ("roundabout", "rotary", "roundabout turn"):
        return f"At the roundabout take exit {m.get('exit', 1)}{on}"
    if mod == "uturn":
        return "Make a U-turn"
    if mod in ("", "straight"):
        return f"Continue straight{on}"
    if mod.startswith("slight"):
        return f"Bear {mod.split()[-1]}{on}"
    if mod.startswith("sharp"):
        return f"Take a sharp {mod.split()[-1]}{on}"
    return f"Turn {mod}{on}"


def osrm_route(a_lat, a_lon, b_lat, b_lon):
    url = f"{OSRM_URL}/route/v1/driving/{a_lon},{a_lat};{b_lon},{b_lat}"
    r = requests.get(url, params={"overview": "full", "geometries": "geojson", "steps": "true"},
                     headers={"User-Agent": UA}, timeout=25)
    j = r.json()
    if j.get("code") != "Ok":
        return None
    rt = j["routes"][0]
    coords = [(c[1], c[0]) for c in rt["geometry"]["coordinates"]]
    steps = []
    for leg in rt["legs"]:
        for s in leg["steps"]:
            m = s["maneuver"]
            steps.append({"lat": m["location"][1], "lon": m["location"][0],
                          "text": instruction(m, s.get("name", "")), "type": m.get("type", "")})
    return {"coords": coords, "steps": steps, "distance": rt["distance"], "duration": rt["duration"]}


def save_route(dev, r, dlat, dlon, dname):
    coords = r["coords"]
    cum = [0.0]
    for i in range(1, len(coords)):
        cum.append(cum[-1] + hav(coords[i - 1][0], coords[i - 1][1], coords[i][0], coords[i][1]))
    prev = 0
    for s in r["steps"]:
        s["idx"], _ = nearest_idx(coords, s["lat"], s["lon"], prev, len(coords))
        prev = s["idx"]
    ROUTES[dev] = {"coords": coords, "cum": cum, "steps": r["steps"], "distance": r["distance"],
                   "duration": r["duration"], "dest": (dlat, dlon, dname), "last": 0}


# ----------------------------------------------------------------- voice
def process_voice(wav: bytes, lat: float, lon: float, dev: str):
    if len(wav) < 2000:
        return {"ok": False, "say": "I did not hear anything."}
    try:
        heard = transcribe(wav)
    except Exception as e:
        print("STT error:", e)
        return {"ok": False, "say": "The speech service had an error."}
    if not heard:
        return {"ok": False, "heard": "", "say": "Sorry, I could not understand that."}
    query = extract_destination(heard)
    try:
        g = geocode(query, lat, lon)
    except Exception as e:
        print("geocode error:", e)
        g = None
    if not g:
        return {"ok": False, "heard": heard, "say": f"I could not find {query}."}
    glat, glon, gname = g
    try:
        r = osrm_route(lat, lon, glat, glon)
    except Exception as e:
        print("route error:", e)
        r = None
    if not r:
        return {"ok": False, "heard": heard, "say": "I could not find a driving route there."}
    save_route(dev, r, glat, glon, gname)
    short = gname.split(",")[0]
    km, mins = r["distance"] / 1000, round(r["duration"] / 60)
    return {"ok": True, "heard": heard, "dest": short, "left_m": int(r["distance"]),
            "eta_s": int(r["duration"]),
            "say": f"Route to {short} found. {km:.1f} kilometres, about {mins} minutes."}


@app.get("/hello")
def hello(x_device_token: str = Header(default="")):
    check(x_device_token)
    return {"ok": True, "t": int(time.time())}


@app.post("/voice")
async def voice(request: Request, lat: float, lon: float, dev: str = "dev",
                x_device_token: str = Header(default="")):
    check(x_device_token)
    wav = await request.body()
    return await run_in_threadpool(process_voice, wav, lat, lon, dev)


# ----------------------------------------------------------------- navigation
@app.get("/nav/update")
def nav_update(lat: float, lon: float, dev: str = "dev", x_device_token: str = Header(default="")):
    check(x_device_token)
    R = ROUTES.get(dev)
    if not R:
        return {"state": "none"}
    coords, cum = R["coords"], R["cum"]
    dlat, dlon, dname = R["dest"]

    if hav(lat, lon, dlat, dlon) < 30:
        ROUTES.pop(dev, None)
        return {"state": "arrived"}

    last = R["last"]
    idx, d = nearest_idx(coords, lat, lon, max(0, last - 5), last + 400)
    if d > 60:
        idx, d = nearest_idx(coords, lat, lon, 0, len(coords))
    rerouted = False
    if d > 60:
        try:
            r = osrm_route(lat, lon, dlat, dlon)
        except Exception as e:
            print("reroute error:", e)
            r = None
        if r:
            save_route(dev, r, dlat, dlon, dname)
            R = ROUTES[dev]
            coords, cum = R["coords"], R["cum"]
            idx, d = 0, 0.0
            rerouted = True
    R["last"] = idx

    steps = R["steps"]
    nxt = len(steps) - 1
    for i, s in enumerate(steps):
        if i == 0 and len(steps) > 1:
            continue
        if s["idx"] >= idx:
            nxt = i
            break
    s = steps[nxt]
    dist = max(0, cum[s["idx"]] - cum[idx] + d)
    left = max(0.0, cum[-1] - cum[idx])
    eta = int(R["duration"] * left / max(1.0, R["distance"]))
    return {"state": "nav", "step": nxt, "text": s["text"], "dist": int(dist),
            "left_m": int(left), "eta_s": eta, "rerouted": rerouted}


@app.post("/nav/cancel")
def nav_cancel(dev: str = "dev", x_device_token: str = Header(default="")):
    check(x_device_token)
    ROUTES.pop(dev, None)
    return {"ok": True}


# ----------------------------------------------------------------- map rendering
@lru_cache(maxsize=256)
def get_tile(z, x, y):
    n = 2 ** z
    if y < 0 or y >= n:
        return Image.new("RGB", (256, 256), (225, 225, 225))
    x %= n
    try:
        r = requests.get(TILE_URL.format(z=z, x=x, y=y), headers={"User-Agent": UA}, timeout=10)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception as e:
        print("tile error:", e)
        return Image.new("RGB", (256, 256), (225, 225, 225))


def render_map(lat, lon, z, plat, plon, dev):
    cx, cy = latlon_to_px(lat, lon, z)
    x0, y0 = cx - W / 2, cy - H / 2
    tx0, tx1 = int(x0 // 256), int((x0 + W) // 256)
    ty0, ty1 = int(y0 // 256), int((y0 + H) // 256)
    canvas = Image.new("RGB", ((tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256))
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            canvas.paste(get_tile(z, tx, ty), ((tx - tx0) * 256, (ty - ty0) * 256))
    ox, oy = int(x0 - tx0 * 256), int(y0 - ty0 * 256)
    img = canvas.crop((ox, oy, ox + W, oy + H))
    d = ImageDraw.Draw(img)

    R = ROUTES.get(dev)
    if R:
        pts = []
        for la, lo in R["coords"]:
            px, py = latlon_to_px(la, lo, z)
            pts.append((px - x0, py - y0))
        if len(pts) >= 2:
            d.line(pts, fill=(255, 255, 255), width=6, joint="curve")
            d.line(pts, fill=(0, 90, 255), width=3, joint="curve")
        dx_, dy_ = latlon_to_px(R["dest"][0], R["dest"][1], z)
        dx_, dy_ = dx_ - x0, dy_ - y0
        d.rectangle([dx_ - 5, dy_ - 5, dx_ + 5, dy_ + 5], fill=(0, 170, 60), outline=(255, 255, 255))

    me_lat = plat if plat is not None else lat
    me_lon = plon if plon is not None else lon
    mx, my = latlon_to_px(me_lat, me_lon, z)
    mx, my = mx - x0, my - y0
    d.ellipse([mx - 6, my - 6, mx + 6, my + 6], fill=(255, 255, 255))
    d.ellipse([mx - 4, my - 4, mx + 4, my + 4], fill=(220, 0, 0))

    a = np.asarray(img).astype(np.uint16)
    rgb565 = ((a[..., 0] & 0xF8) << 8) | ((a[..., 1] & 0xFC) << 3) | (a[..., 2] >> 3)
    return rgb565.astype("<u2").tobytes()


@app.get("/map")
def map_image(lat: float, lon: float, z: int = 16, dev: str = "dev",
              plat: float | None = None, plon: float | None = None,
              x_device_token: str = Header(default="")):
    check(x_device_token)
    z = max(10, min(18, z))
    try:
        data = render_map(lat, lon, z, plat, plon, dev)
    except Exception as e:
        print("map error:", e)
        raise HTTPException(status_code=502, detail="map render failed")
    return Response(content=data, media_type="application/octet-stream")


# ----------------------------------------------------------------- text to speech
def tts_pcm(text: str) -> bytes:
    mp3 = io.BytesIO()
    gTTS(text=text, lang="en", tld="co.in").write_to_fp(mp3)
    p = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-ac", "1", "-ar", "8000", "-f", "u8", "pipe:1"],
        input=mp3.getvalue(), capture_output=True, check=True)
    return p.stdout[:80000]


@app.get("/tts")
def tts(text: str, x_device_token: str = Header(default="")):
    check(x_device_token)
    text = text.strip()[:160]
    if text not in TTS_CACHE:
        try:
            TTS_CACHE[text] = tts_pcm(text)
        except Exception as e:
            print("tts error:", e)
            raise HTTPException(status_code=502, detail="tts failed")
        if len(TTS_CACHE) > 200:
            TTS_CACHE.pop(next(iter(TTS_CACHE)))
    return Response(content=TTS_CACHE[text], media_type="application/octet-stream")
