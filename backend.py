from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import httpx
import os
from pathlib import Path

app = FastAPI(title="CardioAI Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

KAGGLE_URL = "https://vendetta-twiddle-repose.ngrok-free.dev"

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = Path(__file__).parent / "cardiac_frontend.html"
    if html_path.exists():
        content = html_path.read_text(encoding="utf-8", errors="replace")
        return HTMLResponse(content=content, status_code=200)
    return HTMLResponse("<h2>Frontend not found. Place cardiac_frontend.html next to backend.py</h2>")

@app.get("/health")
async def health():
    return {"status": "ok", "kaggle_url_configured": bool(KAGGLE_URL)}

@app.post("/predict")
async def predict(request: Request):
    body = await request.json()
    kaggle_url = (
        request.headers.get("x-kaggle-url") or KAGGLE_URL
    ).rstrip("/")

    if not kaggle_url:
        raise HTTPException(status_code=400, detail="No Kaggle URL configured.")

    target = f"{kaggle_url}/predict"

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                target,
                json=body,
                headers={
                    "Content-Type": "application/json",
                    "ngrok-skip-browser-warning": "true",
                }
            )
            response.raise_for_status()
            return JSONResponse(content=response.json())
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Kaggle kernel timed out (60s).")
    except httpx.ConnectError as e:
        raise HTTPException(status_code=502, detail=f"Cannot reach Kaggle URL: {str(e)}")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
