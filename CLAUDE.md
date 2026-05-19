# Vision Navigator — CLAUDE.md

## Project overview
Mobile-web AR indoor navigation system for SZABIST 100 Campus, Karachi.
Backend: Python Flask + DINOv2 + FAISS + A*.  Frontend: vanilla HTML5/ES Modules/Canvas.

## Directory layout
```
szabnavai/
├── app.py               ← CampusNavigationApp (DINOv2 + FAISS inference)
├── flask_api.py         ← Flask REST API (port 5000)
├── pathfinding.py       ← CampusGraph + AStarPathfinder  ← only .py to meaningfully edit
├── embedding_model.py   ← DINOv2Embedder
├── faiss_index.py       ← FAISSIndex (FAISS IndexFlatL2)
├── config.py            ← all constants (API_VERSION, CORS_ORIGINS, paths)
├── data_loader.py       ← dataset loader (train_index.py only)
├── train_index.py       ← admin: build FAISS index from dataset/
├── campus.index         ← FAISS index (binary, gitignored)
├── labels.npy           ← label array (gitignored)
├── metadata.json        ← index metadata
├── zabgpt/
│   ├── src/rag.py       ← VisionNavigator RAG class
│   └── src/config.py   ← ZabGPT constants (Chroma path, LLM config)
└── web/
    ├── index.html       ← Home screen
    ├── destination.html ← Destination picker
    ├── ar.html          ← AR camera screen
    ├── calibrate.html   ← Calibration pan screen
    ├── css/styles.css   ← shared styles + brand tokens
    ├── js/
    │   ├── api.js          ← fetch wrappers (all endpoints)
    │   ├── state.js        ← sessionStorage app state
    │   ├── camera.js       ← getUserMedia + JPEG capture
    │   ├── sensors.js      ← DeviceOrientation + iOS permission
    │   ├── heading.js      ← relative-heading state machine
    │   ├── calibrate.js    ← calibration pan logic
    │   ├── ar-overlay.js   ← Canvas arrow renderer
    │   ├── minimap.js      ← Canvas mini-map renderer
    │   ├── main-home.js    ← home page bootstrap
    │   └── main-ar.js      ← AR orchestrator
    └── assets/
        ├── floorplan.svg
        └── icon-arrow.svg
```

## Running the backend
```powershell
cd C:\Users\prime\projects\szabnavai
pip install -r requirements.txt
python flask_api.py        # starts on http://localhost:5000
```
ZabGPT chroma_db must be present at `C:\Users\prime\Desktop\zabgpt\chroma_db\`
(path configured in `zabgpt/src/config.py`).

## Campus nodes (6 — matches trained FAISS index)
| node_id      | label              | floor |
|--------------|--------------------|-------|
| Cafe         | Canteen            | 0     |
| Courtyard    | Courtyard          | 0     |
| FountainArea | Fountain Area      | 0     |
| helmetarea   | Helmet Area        | 0     |
| SSC          | SSC Room           | 1     |
| stairs       | Main Gate/100 Door | 0     |

Coordinates (metres, origin = Main Gate) live in `NODE_COORDS` at top of `pathfinding.py`.
Edit that dict when real survey data is available.

## API endpoints (Flask :5000)
| Method | Path | Purpose |
|--------|------|---------|
| GET | /api/health | Liveness + model status |
| GET | /api/locations | All nodes with coords + neighbours |
| POST | /api/identify | Image → node_id + coordinates |
| POST | /api/navigate | from_node + to_node → path + waypoints |
| POST | /api/full-navigate | Image + to_node → full result |
| POST | /api/calibrate | N frames + alphas → anchor calibration |
| POST | /api/zabgpt | query → RAG answer |

## Frontend architecture
- Multi-page (no SPA router). Pages share state via `sessionStorage` (managed in `state.js`).
- AR heading is gyroscope-relative only — no compass dependency.
- Re-localisation fires every 3500 ms via `setInterval` in `main-ar.js`.
- Calibration is required at the start of every AR session.

## Key rules (do not break)
- Never modify DINOv2 / FAISS / A* core logic — only ADD to responses and ADD new methods.
- In flask_api.py: no disk writes for uploaded images — use `PIL.Image.open(io.BytesIO(...))`.
- Frontend stack: HTML5 + ES Modules + Canvas 2D only. No React, no Vite, no TypeScript.
- JPEG captures: quality 0.7, max 1280×720, ≤ 2 MB per frame.
