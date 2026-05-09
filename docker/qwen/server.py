import io
import base64
import gc
import tempfile
import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI(title="Qwen VLM Service")

model_state = {
    "model": None,
    "tokenizer": None,
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
    if model_state["model"] is not None:
        del model_state["model"]
        del model_state["tokenizer"]
        model_state["model"] = None
        model_state["tokenizer"] = None
        gc.collect()
        torch.cuda.empty_cache()
    
    try:
        print(f"Loading Qwen model from {req.model_path}")
        tokenizer = AutoTokenizer.from_pretrained(req.model_path, trust_remote_code=True)
        
        try:
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            model = AutoModelForCausalLM.from_pretrained(
                req.model_path, 
                device_map="cuda", 
                trust_remote_code=True,
                quantization_config=quant_config
            ).eval()
        except Exception as e:
            print(f"Failed to load with 4-bit, falling back to FP16: {e}")
            model = AutoModelForCausalLM.from_pretrained(
                req.model_path, 
                device_map="cuda", 
                trust_remote_code=True, 
                torch_dtype=torch.float16
            ).eval()
            
        model_state["model"] = model
        model_state["tokenizer"] = tokenizer
        model_state["loaded_path"] = req.model_path
        return {"status": "success"}
    except Exception as e:
        print(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
def generate(req: GenerateRequest):
    if model_state["model"] is None:
        raise HTTPException(status_code=400, detail="Model not loaded")
    
    try:
        image_data = base64.b64decode(req.image_base64)
        pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            pil_image.save(tmp_path)
            
        try:
            query = model_state["tokenizer"].from_list_format([
                {'image': tmp_path},
                {'text': req.prompt},
            ])
            
            with torch.inference_mode():
                response, _ = model_state["model"].chat(model_state["tokenizer"], query=query, history=None)
                
            return {"caption": response.strip()}
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    except Exception as e:
        print(f"Error during generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/unload")
def unload():
    if model_state["model"] is not None:
        del model_state["model"]
        del model_state["tokenizer"]
        model_state["model"] = None
        model_state["tokenizer"] = None
        model_state["loaded_path"] = None
        gc.collect()
        torch.cuda.empty_cache()
    return {"status": "unloaded"}
