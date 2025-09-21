"""
Inbound Carrier Sales API
-------------------------

Purpose:
- Provide a tiny backend for the HappyRobot FDE challenge that:
  1) Verifies carriers against FMCSA (proxying the public endpoint).
  2) Searches a load CSV (hosted in GCS) with strict filters used by the agent.

Highlights:
- Clean request/response models with Pydantic.
- Simple API-key auth via `X-API-Key`.
- CSV loaded into memory with light normalization for stable filtering.
- Deterministic ranking; returns up to 5 formatted "best_option_*" results.
- Debug endpoints (/debug/*) to help demo & troubleshoot.

Security notes:
- HTTPS should be terminated by your cloud entrypoint (e.g., Cloud Run / ALB / Nginx).
- Keep `API_KEY` and `FMCSA_KEY` out of source; pass via environment variables.
"""

from __future__ import annotations

import os
import logging
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime

import pandas as pd
import httpx
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# App & Settings
# -----------------------------------------------------------------------------

app = FastAPI(
    title="Inbound Carrier Sales API",
    version="1.0.0",
    description="Tiny backend for the HappyRobot FDE challenge.",
)

# Environment-driven configuration
API_KEY: str = os.getenv("API_KEY", "supersecretapikey")
LOADS_CSV_URL: Optional[str] = os.getenv("LOADS_CSV_URL")  # e.g. https://storage.googleapis.com/.../loads.csv
FMCSA_KEY: Optional[str] = os.getenv("FMCSA_KEY")          # FMCSA webKey

# CORS (handy for local demo tools / web UIs)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # narrow this for production
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("broker-api")

# -----------------------------------------------------------------------------
# Data model (requests / responses)
# -----------------------------------------------------------------------------

class VerifyCarrierIn(BaseModel):
    """Request body for /verify-carrier."""
    mc_number: str = Field(..., description="FMCSA MC number (digits only)")

class VerifyCarrierOut(BaseModel):
    MCNumber: str
    AllowedToOperate: str
    LegalName: str
    USDOT: str

class SearchLoadsIn(BaseModel):
    """Request body for /loads/search."""
    equipment_type_pref: str = Field(..., description="One of: Dry Van | Reefer | Flatbed")
    origin_pref: str = Field(..., description="Exact 'City, ST' pair (e.g., 'Dallas, TX')")
    destination_pref: Optional[str] = Field("any", description="'any' or exact 'City, ST'")
    pickup_earliest_pref: Optional[str] = Field(None, description="ISO8601 w/ offset (optional)")
    pickup_latest_pref: Optional[str] = Field(None, description="ISO8601 w/ offset (optional)")
    top_k: int = Field(5, ge=1, le=5, description="Max results (1..5)")

class BestOption(BaseModel):
    load_id: str
    origin: str
    destination: str
    pickup_datetime: str
    delivery_datetime: str
    equipment_type: str
    miles: int
    loadboard_rate: float
    rpm: float
    weight: int
    commodity_type: str
    notes: str

class SearchLoadsOut(BaseModel):
    best_option_1: Optional[BestOption | str] = ""
    best_option_2: Optional[BestOption | str] = ""
    best_option_3: Optional[BestOption | str] = ""
    best_option_4: Optional[BestOption | str] = ""
    best_option_5: Optional[BestOption | str] = ""
    debug_counts: Dict[str, int]

# -----------------------------------------------------------------------------
# In-memory CSV store & helpers
# -----------------------------------------------------------------------------

loads_df: Optional[pd.DataFrame] = None

BASE_COLUMNS = [
    "load_id","origin","destination","pickup_datetime","delivery_datetime",
    "equipment_type","loadboard_rate","notes","weight","commodity_type",
    "num_of_pieces","miles","dimensions"
]

_VALID_EQUIPMENT = {"dry van", "reefer", "flatbed"}

def require_api_key(x_api_key: Optional[str]) -> None:
    """Simple header-based API key check."""
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")

def _normalize_equipment(val: Any) -> str:
    if val is None:
        return ""
    s = str(val).strip().lower()
    if "dry" in s and "van" in s:
        return "Dry Van"
    if "reefer" in s or "refrigerated" in s:
        return "Reefer"
    if "flat" in s and "bed" in s:
        return "Flatbed"
    if s in {"dry van", "reefer", "flatbed"}:
        return s.title()
    # fallback: title-case whatever got passed (keeps data visible)
    return s.title()

def _parse_iso(dt: Optional[str]) -> Optional[datetime]:
    """Parse ISO8601 (with/without timezone) to a timezone-aware UTC datetime (as naive)."""
    if not dt or pd.isna(dt):
        return None
    try:
        # pandas returns tz-aware UTC; convert to python datetime and drop tzinfo to compare uniformly
        return pd.to_datetime(dt, utc=True).to_pydatetime().replace(tzinfo=None)
    except Exception:
        return None

def _split_city_state(s: str) -> Tuple[str, str]:
    """'City, ST' -> ('City','ST'). If no comma, returns (trimmed, '')."""
    if not isinstance(s, str):
        return "", ""
    parts = [p.strip() for p in s.split(",")]
    if len(parts) >= 2:
        return parts[0], parts[1]
    return s.strip(), ""

def _split_range_or_single(s: str) -> Tuple[str, str]:
    """Handles either a single ISO or a range separated by en dash '–'."""
    if not isinstance(s, str) or not s:
        return "", ""
    if "–" in s:
        a, b = s.split("–", 1)
        return a.strip(), b.strip()
    return s.strip(), s.strip()

def _city_state_from(pref: str) -> Optional[Tuple[str, str]]:
    """Strict 'City, ST' to tuple; returns None if malformed."""
    if not pref:
        return None
    parts = [p.strip() for p in pref.split(",")]
    if len(parts) != 2:
        return None
    city, state = parts[0], parts[1]
    if not city or not state:
        return None
    return city, state

def _window_overlaps(a_start: Optional[datetime], a_end: Optional[datetime],
                     b_start: Optional[datetime], b_end: Optional[datetime]) -> bool:
    """
    Return True if time ranges [a_start, a_end] and [b_start, b_end] overlap.
    Any None bound means 'open'.
    """
    if b_start is None and b_end is None:
        return True
    a0 = a_start or datetime.min
    a1 = a_end or datetime.max
    b0 = b_start or datetime.min
    b1 = b_end or datetime.max
    return (a0 <= b1) and (b0 <= a1)

def _format_iso_range(start_raw: str, end_raw: str) -> str:
    """Return 'start–end' if different; otherwise a single ISO string."""
    s = start_raw or ""
    e = end_raw or ""
    if s and e and s != e:
        return f"{s}–{e}"
    return s or e

def _rank_key(row: Dict[str, Any]) -> tuple:
    """
    Sort ascending by pickup time, then descending RPM, then descending rate, then ascending miles.
    Lower tuple values come first; we negate for DESC fields.
    """
    ps = row.get("pickup_start_dt")
    rpm = row.get("rpm") if row.get("rpm") is not None else -1e9
    rate = row.get("loadboard_rate") if row.get("loadboard_rate") is not None else -1e9
    miles = row.get("miles") if row.get("miles") is not None else 1e12
    return (ps or datetime.max, -rpm, -rate, miles)

def _format_row(r: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """Format a raw data record into the exact BestOption response shape."""
    origin = r.get("origin") or f"{r.get('origin_city','')}, {r.get('origin_state','')}".strip().strip(",")
    destination = r.get("destination") or f"{r.get('destination_city','')}, {r.get('destination_state','')}".strip().strip(",")
    pickup_str = _format_iso_range(r.get("pickup_start",""), r.get("pickup_end",""))
    delivery_str = _format_iso_range(r.get("delivery_start",""), r.get("delivery_end",""))
    equipment_out = r.get("equipment_type","") or r.get("equipment_type_norm","")
    miles_val = r.get("miles")
    rate_val  = r.get("loadboard_rate")
    weight_val= r.get("weight")
    rpm_val   = r.get("rpm")
    return {
        "load_id": r.get("load_id") or r.get("id") or f"row-{idx}",
        "origin": origin,
        "destination": destination,
        "pickup_datetime": pickup_str,
        "delivery_datetime": delivery_str,
        "equipment_type": equipment_out,
        "miles": int(miles_val) if pd.notna(miles_val) else 0,
        "loadboard_rate": float(rate_val) if pd.notna(rate_val) else 0.0,
        "rpm": round(float(rpm_val), 2) if rpm_val is not None and pd.notna(rpm_val) else 0.0,
        "weight": int(weight_val) if pd.notna(weight_val) else 0,
        "commodity_type": r.get("commodity_type",""),
        "notes": r.get("notes","")
    }

def load_csv_into_memory() -> None:
    """
    Load CSV from LOADS_CSV_URL and derive helper columns:
    - equipment_type_norm
    - origin_city/state, destination_city/state
    - pickup_start/end (and *_dt), delivery_start/end (and *_dt)
    """
    global loads_df
    try:
        if not LOADS_CSV_URL:
            raise RuntimeError("LOADS_CSV_URL not set")

        df = pd.read_csv(LOADS_CSV_URL, dtype=str)

        # Ensure all base columns exist
        for col in BASE_COLUMNS:
            if col not in df.columns:
                df[col] = ""

        # Coerce numerics
        for col in ["miles", "loadboard_rate", "weight"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Normalize equipment
        df["equipment_type_norm"] = df["equipment_type"].apply(_normalize_equipment)

        # Split city/state
        o_cs = df["origin"].apply(_split_city_state)
        d_cs = df["destination"].apply(_split_city_state)
        df["origin_city"] = o_cs.apply(lambda t: t[0])
        df["origin_state"] = o_cs.apply(lambda t: t[1])
        df["destination_city"] = d_cs.apply(lambda t: t[0])
        df["destination_state"] = d_cs.apply(lambda t: t[1])

        # Handle single or range (en dash '–')
        p_range = df["pickup_datetime"].apply(_split_range_or_single)
        d_range = df["delivery_datetime"].apply(_split_range_or_single)
        df["pickup_start"] = p_range.apply(lambda t: t[0])
        df["pickup_end"]   = p_range.apply(lambda t: t[1])
        df["delivery_start"] = d_range.apply(lambda t: t[0])
        df["delivery_end"]   = d_range.apply(lambda t: t[1])

        # Parse to datetime (as naive UTC for uniform comparisons)
        for col in ["pickup_start","pickup_end","delivery_start","delivery_end"]:
            df[col + "_dt"] = df[col].apply(_parse_iso)

        # Trim strings
        for col in ["origin_city","origin_state","destination_city","destination_state"]:
            df[col] = df[col].astype(str).str.strip()

        loads_df = df
        log.info("Loaded %s rows (derived columns ready).", len(loads_df))
    except Exception as e:
        log.exception("Failed to load CSV: %s", e)
        loads_df = pd.DataFrame()

# Load once at startup
load_csv_into_memory()

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.get("/healthz", tags=["meta"])
def healthz() -> Dict[str, str]:
    """Liveness check used by orchestrators & smoke tests."""
    return {"status": "ok"}

@app.post("/verify-carrier", response_model=VerifyCarrierOut, tags=["carrier"])
def verify_carrier(body: VerifyCarrierIn, x_api_key: Optional[str] = Header(default=None)):
    """Proxy the FMCSA Quick Check by MC number and return a compact summary."""
    require_api_key(x_api_key)
    if not FMCSA_KEY:
        raise HTTPException(status_code=500, detail="FMCSA_KEY not configured")

    url = f"https://mobile.fmcsa.dot.gov/qc/services/carriers/docket-number/{body.mc_number}?webKey={FMCSA_KEY}"
    try:
        r = httpx.get(url, timeout=10.0)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"FMCSA upstream error: {e}")

    # The FMCSA payload has multiple shapes; unify it here.
    carrier = {}
    if isinstance(data, dict):
        if "content" in data and isinstance(data["content"], list) and data["content"]:
            first = data["content"][0]
            if isinstance(first, dict):
                carrier = first.get("carrier", {}) or {}
        if not carrier:
            carrier = data.get("carrier", {}) or {}

    allowed = carrier.get("allowedToOperate", "N")
    legal = carrier.get("legalName") or carrier.get("dbaName") or ""
    usdot = carrier.get("usdotNumber") or carrier.get("usdot") or carrier.get("usdDot") or ""

    return {"MCNumber": body.mc_number, "AllowedToOperate": allowed, "LegalName": legal, "USDOT": usdot}

# ---- Debug utilities (handy for the demo; keep them!) -----------------------

@app.get("/debug/schema", tags=["debug"])
def debug_schema(x_api_key: Optional[str] = Header(default=None)):
    """Show basic dataset info and a few example origin/destination pairs."""
    require_api_key(x_api_key)
    if loads_df is None:
        return {"loaded": False}
    origin_examples = (loads_df["origin_city"].astype(str) + ", " + loads_df["origin_state"].astype(str)).dropna().unique().tolist()
    dest_examples = (loads_df["destination_city"].astype(str) + ", " + loads_df["destination_state"].astype(str)).dropna().unique().tolist()
    return {
        "loaded": True,
        "rows": int(len(loads_df)),
        "columns": list(loads_df.columns),
        "sample_header": {c: str(loads_df[c].iloc[0]) if len(loads_df) else "" for c in loads_df.columns[:12]},
        "distinct_equipment_type_norm": sorted(loads_df["equipment_type_norm"].dropna().unique().tolist()),
        "origin_examples": sorted([x for x in origin_examples if x.strip() and x != ", "])[:10],
        "destination_examples": sorted([x for x in dest_examples if x.strip() and x != ", "])[:10]
    }

@app.get("/debug/reload", tags=["debug"])
def debug_reload(x_api_key: Optional[str] = Header(default=None)):
    """Force reload of the CSV from GCS (use after replacing the file)."""
    require_api_key(x_api_key)
    load_csv_into_memory()
    return {"reloaded": True, "rows": int(len(loads_df)) if loads_df is not None else 0}

@app.get("/debug/peek", tags=["debug"])
def debug_peek(n: int = Query(5, ge=1, le=50), x_api_key: Optional[str] = Header(default=None)):
    """Return the first N raw rows for quick inspection."""
    require_api_key(x_api_key)
    if loads_df is None:
        return {"rows": 0, "data": []}
    return {"rows": int(min(n, len(loads_df))), "data": loads_df.head(n).to_dict(orient="records")}

@app.get("/debug/stats", tags=["debug"])
def debug_stats(x_api_key: str | None = Header(default=None)):
    """Counts by origin and by (origin, equipment) to pick good demo scenarios."""
    require_api_key(x_api_key)
    if loads_df is None or loads_df.empty:
        return {"rows": 0, "by_origin": [], "by_origin_equipment": []}
    df = loads_df.copy()
    df["origin_pair"] = (df["origin_city"].astype(str).str.strip() + ", " + df["origin_state"].astype(str).str.strip())
    by_origin = (df.groupby("origin_pair", dropna=False).size()
                   .reset_index(name="count")
                   .sort_values("count", ascending=False)
                   .to_dict(orient="records"))
    by_origin_eq = (df.groupby(["origin_pair", "equipment_type_norm"], dropna=False).size()
                      .reset_index(name="count")
                      .sort_values("count", ascending=False)
                      .to_dict(orient="records"))
    return {"rows": int(len(df)), "by_origin": by_origin, "by_origin_equipment": by_origin_eq}

# ---- Core search ------------------------------------------------------------

@app.post("/loads/search", response_model=SearchLoadsOut, tags=["loads"])
def loads_search(body: SearchLoadsIn, x_api_key: Optional[str] = Header(default=None)):
    """
    Search loads by:
    - equipment_type_pref (exact in {Dry Van, Reefer, Flatbed}, case-insensitive),
    - origin_pref (exact 'City, ST'),
    - optional destination_pref ('any' or exact 'City, ST'),
    - optional pickup window overlap.

    Returns up to 5 best options ranked by:
    pickup time ASC, then RPM DESC, then rate DESC, then miles ASC.
    """
    require_api_key(x_api_key)

    # Ensure data loaded
    if loads_df is None or loads_df.empty:
        load_csv_into_memory()
    df = loads_df.copy()
    counters = {"start": int(len(df))}

    # Equipment (normalized, exact)
    eq = (body.equipment_type_pref or "").strip().lower()
    if eq not in _VALID_EQUIPMENT:
        raise HTTPException(status_code=400, detail="equipment_type_pref must be one of: Dry Van, Reefer, Flatbed")
    eq_title = eq.title()
    df = df[df["equipment_type_norm"] == eq_title]
    counters["after_equipment"] = int(len(df))

    # Origin exact "City, ST"
    origin_pair = _city_state_from(body.origin_pref or "")
    if not origin_pair:
        raise HTTPException(status_code=400, detail="origin_pref must be 'City, ST' (e.g., 'Dallas, TX')")
    o_city, o_state = origin_pair
    df = df[
        (df["origin_city"].str.casefold() == o_city.casefold()) &
        (df["origin_state"].str.casefold() == o_state.casefold())
    ]
    counters["after_origin"] = int(len(df))

    # Destination exact unless 'any'
    dest_pref = (body.destination_pref or "").strip()
    if dest_pref and dest_pref.lower() != "any":
        d_pair = _city_state_from(dest_pref)
        if not d_pair:
            raise HTTPException(status_code=400, detail="destination_pref must be 'City, ST' or 'any'")
        d_city, d_state = d_pair
        df = df[
            (df["destination_city"].str.casefold() == d_city.casefold()) &
            (df["destination_state"].str.casefold() == d_state.casefold())
        ]
    counters["after_destination"] = int(len(df))

    # Pickup window overlap (optional)
    earliest = _parse_iso(body.pickup_earliest_pref) if body.pickup_earliest_pref else None
    latest   = _parse_iso(body.pickup_latest_pref)   if body.pickup_latest_pref   else None
    if earliest or latest:
        mask = df.apply(
            lambda r: _window_overlaps(r.get("pickup_start_dt"), r.get("pickup_end_dt"), earliest, latest),
            axis=1
        )
        df = df[mask]
    counters["after_pickup_window"] = int(len(df))

    # No matches → return empties + counters (kept for demo clarity)
    if df.empty:
        resp = {f"best_option_{i}": "" for i in range(1, 6)}
        resp["debug_counts"] = counters
        return resp  # FastAPI will validate against response_model

    # Compute RPM safely
    miles_num = pd.to_numeric(df["miles"], errors="coerce")
    rate_num  = pd.to_numeric(df["loadboard_rate"], errors="coerce")
    rpm_series = (rate_num / miles_num).where(miles_num > 0, None)
    df = df.assign(rpm=rpm_series)

    # Rank & format top K
    records: List[Dict[str, Any]] = df.to_dict(orient="records")
    records.sort(key=_rank_key)
    top = records[: body.top_k]
    formatted = [_format_row(r, i) for i, r in enumerate(top)]

    # Map to best_option_* with padded empties
    resp: Dict[str, Any] = {f"best_option_{i+1}": (formatted[i] if i < len(formatted) else "") for i in range(5)}
    resp["debug_counts"] = counters
    return resp
