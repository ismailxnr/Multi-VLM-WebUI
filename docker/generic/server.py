import io
import base64
import gc
import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from transformers import pipeline

app = FastAPI(title="Generic VLM Service")

model_state = {
    "pipeline": None,
    "loaded_path": None
}

class LoadRequest(BaseModel):
    model_path: str

class GenerateRequest(BaseModel):
    image_base64: str
    prompt: str

@app.post("/load")
def load_model(req: LoadRequest):
    if model_state["loaded_path"] == req.model_path:
        return {"status": "already loaded"}
    
    # Unload previous
    if model_state["pipeline"] is not None:
        del model_state["pipeline"]
        model_state["pipeline"] = None
        gc.collect()
        torch.cuda.empty_cache()
    
    try:
        print(f"Loading Generic model from {req.model_path}")
        # Default to image-to-text pipeline
        try:
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            pipe = pipeline(
                "image-to-text", 
                model=req.model_path, 
                device_map="auto",
                model_kwargs={"quantization_config": quant_config}
            )
        except Exception as e:
            print(f"Failed to load generic model with 4-bit, falling back to default: {e}")
            pipe = pipeline(
                "image-to-text", 
                model=req.model_path, 
                device_map="auto"
            )
            
        model_state["pipeline"] = pipe
        model_state["loaded_path"] = req.model_path
        return {"status": "success"}
    except Exception as e:
        print(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
def generate(req: GenerateRequest):
    if model_state["pipeline"] is None:
        raise HTTPException(status_code=400, detail="Model not loaded")
    
    try:
        image_data = base64.b64decode(req.image_base64)
        pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        with torch.inference_mode():
            outputs = model_state["pipeline"](pil_image, prompt=req.prompt)
            # Handle different output formats based on the specific model
            if isinstance(outputs, list) and len(outputs) > 0:
                if 'generated_text' in outputs[0]:
                    caption = outputs[0]['generated_text']
                else:
                    caption = str(outputs[0])
            else:
                caption = str(outputs)
                
        return {"caption": caption.strip()}
    except Exception as e:
        print(f"Error during generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/unload")
def unload():
    if model_state["pipeline"] is not None:
        del model_state["pipeline"]
        model_state["pipeline"] = None
        model_state["loaded_path"] = None
        gc.collect()
        torch.cuda.empty_cache()
    return {"status": "unloaded"}
