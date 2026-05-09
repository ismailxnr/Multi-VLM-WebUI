# Multi-Container VLM Web Interface

A scalable, multi-container Docker architecture for running and comparing multiple Vision-Language Models (VLMs) simultaneously without dependency conflicts. 

This project isolates different model families (e.g., Qwen, LLaVA, Generic HuggingFace models) into their own Docker containers, each exposing a simple FastAPI endpoint. A central FastAPI web application acts as a routing layer, directing inference requests to the appropriate container based on the model's family.

## Architecture

- **`webapp` (Port 8000)**: The main user interface and API gateway. It doesn't load any models into VRAM. It simply reads `custom_models.json` and forwards inference requests to the appropriate VLM container over HTTP.
- **`qwen_vlm` (Port 8001)**: Container dedicated to the Qwen family (e.g., `Qwen-VL-Chat`). Requires newer `transformers` versions.
- **`rsllava_vlm` (Port 8002)**: Container dedicated to LLaVA and custom RS-LLaVA models. Uses an older `transformers` version (`4.37.2`) to maintain compatibility with the LLaVA codebase.
- **`generic_vlm` (Port 8003)**: Fallback container for any generic HuggingFace `image-to-text` pipeline.

## Prerequisites

- **NVIDIA GPU**: Required for running the models. The containers use the `nvidia` Docker runtime.
- **Docker Compose**: Required to orchestrate the multi-container setup.

## Setup

1. Clone this repository.
2. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
3. Edit the `.env` file to point to your local directories:
   - `MODELS_DIR`: The directory on your host machine where your model checkpoints are stored. This will be mounted to `/models` inside the containers.
   - `RSLLAVA_DIR`: The path to your local clone of the `LLaVA` or `RS-LLaVA` repository. This is mounted to `/app/RS-LLaVA` in the `rsllava_vlm` container.
   - `WEB_PORT`: The port on which the web interface will be exposed (default `8000`).

## Running the Application

Build and start the containers using Docker Compose:

```bash
docker compose up --build -d
```

Access the web interface at `http://localhost:8000` (or whatever port you specified in `.env`).

## Managing Models

Models are managed via the web interface. When adding a new custom model through the UI, ensure the `path` you provide is relative to the mount point. 
For example, if your host model is at `/home/user/models/checkpoint-8000` and you set `MODELS_DIR=/home/user/models`, the path you input in the UI should be `/models/checkpoint-8000`.

## Memory Management

The web application is designed with strict memory management. Models are dynamically loaded into VRAM only when needed for an inference request and are immediately unloaded afterward using the `/unload` endpoint of the respective container. This allows you to evaluate multiple massive VLMs sequentially on a single GPU without running into Out-Of-Memory (OOM) errors.
