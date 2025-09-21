# Inbound Carrier Sales API (HappyRobot FDE)

A tiny FastAPI service that:
- verifies carriers via FMCSA,
- searches a load CSV hosted on Google Cloud Storage,
- returns clean, ranked options for an agent to negotiate.

## Endpoints

- `GET /healthz` — liveness check
- `POST /verify-carrier` — body: `{ "mc_number": "1515" }`
- `POST /loads/search` — see **Search Request** below
- `GET /debug/reload` — reload the CSV from `LOADS_CSV_URL`
- `GET /debug/schema` — quick dataset info
- `GET /debug/peek?n=5` — first rows
- `GET /debug/stats` — counts by origin & (origin, equipment)

All endpoints require `X-API-Key`.

### Search Request (example)
```json
{
  "equipment_type_pref": "Flatbed",
  "origin_pref": "Denver, CO",
  "destination_pref": "any",
  "pickup_earliest_pref": "2025-09-22T00:00:00-05:00",
  "pickup_latest_pref":  "2025-09-22T23:59:59-05:00",
  "top_k": 5
}
