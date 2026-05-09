import os
import io
import json
import time
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from model_wrappers import get_wrapper_for_family

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
CUSTOM_MODELS_FILE = os.path.join(os.path.dirname(__file__), "custom_models.json")

app = FastAPI(title="RS-LLaVA Web App")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
model_state = {
    "wrapper": None,
    "loaded_key": None,
}


def load_custom_models():
    """Load user-added custom models from JSON file."""
    if os.path.exists(CUSTOM_MODELS_FILE):
        try:
            with open(CUSTOM_MODELS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading custom models: {e}")
    return {}


def save_custom_model(key, name, path, family):
    """Save a new custom model to the JSON file."""
    custom = load_custom_models()
    custom[key] = {
        "name": name,
        "path": path,
        "family": family,
        "group": "Custom Models",
        "step": 999999
    }
    with open(CUSTOM_MODELS_FILE, "w", encoding="utf-8") as f:
        json.dump(custom, f, indent=4)


def scan_available_models():
    """Load only custom models from JSON file as requested by the user."""
    models = {}

    # Load custom models
    custom_models = load_custom_models()
    for k, v in custom_models.items():
        models[k] = v

    return models


def load_model_by_key(model_key):
    """Load a model checkpoint using the correct wrapper."""
    global model_state
    AVAILABLE_MODELS = scan_available_models()

    if model_state["loaded_key"] == model_key:
        return  # Already loaded

    # Unload previous model remotely
    if model_state["wrapper"] is not None:
        model_state["wrapper"].unload()
        model_state["wrapper"] = None

    if model_key not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model_key} not found.")

    model_info = AVAILABLE_MODELS[model_key]
    model_path = model_info["path"]
    family = model_info.get("family", "rsllava")

    # Instantiate wrapper and load
    wrapper = get_wrapper_for_family(family)
    print(f"Loading {model_key} using {family} wrapper...")
    wrapper.load(model_path)

    model_state["wrapper"] = wrapper
    model_state["loaded_key"] = model_key


# ─── API Endpoints ───────────────────────────────────────────

@app.get("/api/models")
async def list_models():
    """Return available models grouped by training run/custom group."""
    AVAILABLE_MODELS = scan_available_models()
    grouped = {}
    for key, info in AVAILABLE_MODELS.items():
        group = info["group"]
        if group not in grouped:
            grouped[group] = []
        grouped[group].append({"key": key, "name": info["name"], "step": info["step"]})
    return JSONResponse(grouped)


@app.post("/api/models/add")
async def add_custom_model(
    name: str = Form(...),
    path: str = Form(...),
    family: str = Form(...)
):
    """Register a new custom model."""
    try:
        # Use timestamp to ensure unique keys and allow multiple models with same name
        key = f"custom/{name.replace(' ', '_').lower()}_{int(time.time())}"
        save_custom_model(key, name, path, family)
        return JSONResponse({"status": "success", "key": key})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.delete("/api/models/remove")
async def remove_custom_model(key: str = Form(...)):
    """Remove a custom model from the database."""
    try:
        custom_models = load_custom_models()
        if key in custom_models:
            del custom_models[key]
            with open(CUSTOM_MODELS_FILE, "w", encoding="utf-8") as f:
                json.dump(custom_models, f, indent=4)
            return JSONResponse({"status": "success"})
        else:
            return JSONResponse({"status": "error", "message": "Model bulunamadı."}, status_code=404)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/api/compare")
async def compare(image: UploadFile = File(...), models: str = Form(...), prompt: str = Form(None)):
    """Compare multiple models on the same image. Returns results as SSE stream."""
    if not prompt or prompt.strip() == "":
        prompt = "Bu uzaktan algılama görüntüsünü Türkçe olarak açıklayın."

    contents = await image.read()
    pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    model_keys = json.loads(models)
    
    AVAILABLE_MODELS = scan_available_models()

    async def event_stream():
        for i, key in enumerate(model_keys):
            info = AVAILABLE_MODELS.get(key)
            if not info:
                continue

            # Send loading status
            yield f"data: {json.dumps({'type': 'loading', 'model_key': key, 'model_name': info['name'], 'index': i, 'total': len(model_keys)})}\n\n"

            try:
                t0 = time.time()
                load_model_by_key(key)
                load_time = time.time() - t0

                t1 = time.time()
                wrapper = model_state["wrapper"]
                caption = wrapper.generate(pil_image.copy(), prompt)
                infer_time = time.time() - t1

                # Memory strict mode: Unload immediately after inference
                wrapper.unload()
                model_state["wrapper"] = None
                model_state["loaded_key"] = None

                yield f"data: {json.dumps({'type': 'result', 'model_key': key, 'model_name': info['name'], 'caption': caption, 'load_time': round(load_time, 1), 'infer_time': round(infer_time, 1), 'index': i, 'total': len(model_keys)})}\n\n"

            except Exception as e:
                # Ensure memory is freed even on error
                if model_state["wrapper"] is not None:
                    model_state["wrapper"].unload()
                    model_state["wrapper"] = None
                    model_state["loaded_key"] = None
                    
                yield f"data: {json.dumps({'type': 'error', 'model_key': key, 'model_name': info['name'], 'error': str(e), 'index': i, 'total': len(model_keys)})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/generate")
async def generate(image: UploadFile = File(...), prompt: str = Form(None)):
    """Single model inference (backward compatible)."""
    if not prompt or prompt.strip() == "":
        prompt = "Bu uzaktan algılama görüntüsünü Türkçe olarak açıklayın."

    # Load default model if nothing loaded
    if model_state["wrapper"] is None:
        AVAILABLE_MODELS = scan_available_models()
        if not AVAILABLE_MODELS:
            return JSONResponse({"error": "Hiçbir model eklenmemiş. Lütfen önce bir model ekleyin."}, status_code=400)
        default_key = list(AVAILABLE_MODELS.keys())[0]
        load_model_by_key(default_key)

    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        wrapper = model_state["wrapper"]
        caption = wrapper.generate(pil_image, prompt)

        # Memory strict mode
        wrapper.unload()
        model_state["wrapper"] = None
        model_state["loaded_key"] = None

        return JSONResponse({"caption": caption})
    except Exception as e:
        print(f"Error during inference: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# Mount static files
app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static"), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
