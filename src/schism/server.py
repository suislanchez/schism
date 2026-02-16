"""FastAPI server with REST API and WebSocket streaming."""

from __future__ import annotations

import json
import asyncio
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from schism.engine.loader import list_available_models, load_model
from schism.engine.features import get_all_features, get_feature_names
from schism.engine.generate import generate_stream, generate_comparison
from schism.presets.manager import list_presets, get_preset, save_preset, delete_preset

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Schism", version="0.1.0")


# --- REST Endpoints ---


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/models")
async def api_models():
    return list_available_models()


@app.get("/api/features")
async def api_features():
    features = get_all_features()
    return {
        name: {
            "description": feat["description"],
        }
        for name, feat in features.items()
    }


@app.get("/api/presets")
async def api_presets():
    return list_presets()


class PresetCreate(BaseModel):
    name: str
    description: str = ""
    model: str = "any"
    sliders: dict[str, float]


@app.post("/api/presets")
async def api_create_preset(preset: PresetCreate):
    path = save_preset(preset.model_dump())
    return {"status": "ok", "path": str(path)}


@app.delete("/api/presets/{name}")
async def api_delete_preset(name: str):
    success = delete_preset(name)
    if success:
        return {"status": "ok"}
    return JSONResponse({"error": "Preset not found or is a default"}, status_code=404)


@app.post("/api/load-model")
async def api_load_model(body: dict):
    model_name = body.get("model", "gemma-2-2b")
    try:
        load_model(model_name)
        return {"status": "ok", "model": model_name}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# --- WebSocket Streaming ---


@app.websocket("/ws/generate")
async def ws_generate(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_text()
            request = json.loads(data)

            model_name = request.get("model", "gemma-2-2b")
            prompt = request.get("prompt", "")
            sliders = request.get("sliders", {})
            compare = request.get("compare", False)
            max_tokens = request.get("max_tokens", 256)
            temperature = request.get("temperature", 0.7)

            if not prompt:
                await websocket.send_json({"error": "No prompt provided"})
                continue

            # Signal generation start
            await websocket.send_json({"type": "start", "compare": compare})

            try:
                if compare:
                    async for chunk in generate_comparison(
                        model_name, prompt, sliders,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    ):
                        await websocket.send_json({
                            "type": "token",
                            "token": chunk["token"],
                            "side": chunk["side"],
                        })
                        await asyncio.sleep(0)  # yield control
                else:
                    async for token in generate_stream(
                        model_name, prompt, sliders,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    ):
                        await websocket.send_json({
                            "type": "token",
                            "token": token,
                            "side": "steered",
                        })
                        await asyncio.sleep(0)

                await websocket.send_json({"type": "done"})

            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "error": str(e),
                })

    except WebSocketDisconnect:
        pass


# Mount static files last so API routes take priority
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
