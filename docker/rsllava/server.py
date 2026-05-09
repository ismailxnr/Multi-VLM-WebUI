import io
import base64
import gc
import sys
import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image

sys.path.insert(0, "/app/RS-LLaVA")

try:
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates, SeparatorStyle
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
except ImportError as e:
    print(f"Warning: Failed to import llava: {e}")

app = FastAPI(title="RS-LLaVA VLM Service")

model_state = {
    "model": None,
    "tokenizer": None,
    "image_processor": None,
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
        del model_state["image_processor"]
        model_state["model"] = None
        model_state["tokenizer"] = None
        model_state["image_processor"] = None
        gc.collect()
        torch.cuda.empty_cache()
    
    try:
        print(f"Loading RS-LLaVA model from {req.model_path}")
        disable_torch_init()
        model_name = get_model_name_from_path(req.model_path)
        
        offload_dir = "/app/offload_weights"
        os.makedirs(offload_dir, exist_ok=True)
        
        model_base = "liuhaotian/llava-v1.5-7b"
        base = model_base if "checkpoint" in req.model_path or "lora" in req.model_path.lower() else None
        
        tokenizer, model, image_processor, _ = load_pretrained_model(
            req.model_path, base, model_name,
            load_4bit=True,
            offload_folder=offload_dir
        )
        
        model_state["model"] = model
        model_state["tokenizer"] = tokenizer
        model_state["image_processor"] = image_processor
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
        
        target_size = model_state["image_processor"].crop_size.get("height", 336)
        resized_img = pil_image.resize((target_size, target_size), Image.LANCZOS)
        
        image_tensor = model_state["image_processor"].preprocess(resized_img, return_tensors="pt")["pixel_values"][0]

        cur_prompt = f"{DEFAULT_IMAGE_TOKEN}\n{req.prompt}"
        conv = conv_templates["v1"].copy()
        conv.append_message(conv.roles[0], cur_prompt)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            full_prompt, model_state["tokenizer"], IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], model_state["tokenizer"], input_ids)

        with torch.inference_mode():
            output_ids = model_state["model"].generate(
                input_ids,
                images=image_tensor.unsqueeze(0).to(dtype=torch.float16, device='cuda', non_blocking=True),
                do_sample=True,
                temperature=0.5,
                top_p=0.9,
                max_new_tokens=512,
                use_cache=False,
                stopping_criteria=[stopping_criteria],
            )

        input_token_len = input_ids.shape[1]
        outputs = model_state["tokenizer"].batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        return {"caption": outputs.strip()}
    except Exception as e:
        print(f"Error during generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/unload")
def unload():
    if model_state["model"] is not None:
        del model_state["model"]
        del model_state["tokenizer"]
        del model_state["image_processor"]
        model_state["model"] = None
        model_state["tokenizer"] = None
        model_state["image_processor"] = None
        model_state["loaded_path"] = None
        gc.collect()
        torch.cuda.empty_cache()
    return {"status": "unloaded"}
