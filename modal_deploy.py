"""Modal deployment for ACE-Step 1.5 API server on A100.

Speed optimizations applied:
  1. Models baked into image (local SSD, not NFS volume) — eliminates ~30s volume read
  2. Triton cache pre-warmed at build time — avoids JIT compilation on first request
  3. torch.compile disabled — avoids long first-inference compile (A100 is fast enough)

Generated audio is uploaded to GCS bucket studio.diskrot.com.

Setup:
    # Create the Modal secret with your GCS service account key:
    modal secret create gcs-credentials GOOGLE_APPLICATION_CREDENTIALS_JSON='<contents of key.json>'

Usage:
    modal deploy modal_deploy.py
    modal serve modal_deploy.py
"""

import modal

REPO_ID = "ACE-Step/Ace-Step1.5"
MODEL_DIR = "/app/checkpoints"
GPU = "A100-40GB"
GCS_BUCKET = "studio.diskrot.com"

GCS_SECRET = modal.Secret.from_name("gcs-credentials")
API_KEY_SECRET = modal.Secret.from_name("diskrot-api-key")

app = modal.App("acestep")


def _download_models():
    """Download models at image build time — baked into the layer."""
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=REPO_ID,
        local_dir=MODEL_DIR,
        local_dir_use_symlinks=False,
    )


# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        # PyTorch + CUDA 12.8 (Linux x86_64)
        "torch==2.10.0+cu128",
        "torchvision==0.25.0+cu128",
        "torchaudio==2.10.0+cu128",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        # Core deps
        "safetensors==0.7.0",
        "transformers>=4.51.0,<4.58.0",
        "diffusers",
        "scipy>=1.10.1",
        "soundfile>=0.13.1",
        "loguru>=0.7.3",
        "einops>=0.8.1",
        "accelerate>=1.12.0",
        "fastapi>=0.110.0",
        "uvicorn[standard]>=0.27.0",
        "numba>=0.63.1",
        "vector-quantize-pytorch>=1.27.15",
        "torchcodec>=0.9.1",
        "torchao>=0.14.1,<0.16.0",
        "toml",
        "modelscope",
        "peft>=0.18.0",
        "lycoris-lora",
        "lightning>=2.0.0",
        "tensorboard>=2.20.0",
        "typer-slim>=0.21.1",
        "matplotlib>=3.7.5",
        "diskcache",
        "xxhash",
        "huggingface_hub",
        "python-dotenv",
        "triton>=3.0.0",
        # Gradio (needed even for API due to import chains)
        "gradio==6.2.0",
        # GCS upload
        "google-cloud-storage",
    )
    # flash-attn 2.8.3 doesn't support torch 2.10 yet (max 2.9).
    # PyTorch's built-in SDPA uses the same FlashAttention kernels on A100.
    # Copy the entire project into the image
    .add_local_dir(
        ".",
        remote_path="/app",
        copy=True,
        ignore=[
            ".git",
            "__pycache__",
            "*.pyc",
            ".DS_Store",
            "checkpoints",
            ".cache",
            "*.log",
            ".venv",
            "venv",
            "acestep_output",
            "my_dataset",
        ],
    )
    # Install nano-vllm from vendored source and the project itself
    .run_commands(
        "pip install -e /app/acestep/third_parts/nano-vllm",
        "pip install -e /app",
    )
    # ── Key optimisation: bake models into the image layer ──
    # After first build this is cached. Containers read weights from local SSD
    # instead of an NFS volume, shaving ~30s off cold starts.
    .run_function(_download_models)
    # Pre-warm Triton kernel cache so first inference doesn't JIT-compile
    .run_commands(
        "python -c 'import triton; import torch; print(\"triton+torch imported\")'",
    )
    .env(
        {
            "ACESTEP_CONFIG_PATH": "acestep-v15-turbo",
            "ACESTEP_LM_MODEL_PATH": "acestep-5Hz-lm-1.7B",
            "ACESTEP_LM_BACKEND": "vllm",
            "ACESTEP_INIT_LLM": "true",
            "ACESTEP_DEVICE": "cuda",
            "ACESTEP_DOWNLOAD_SOURCE": "huggingface",
            "ACESTEP_API_HOST": "0.0.0.0",
            "ACESTEP_API_PORT": "8001",
            "ACESTEP_NO_INIT": "false",
            # Disable torch.compile — avoids multi-minute first-inference warmup.
            # A100 is fast enough without it; the cold-start savings are worth more.
            "ACESTEP_COMPILE_MODEL": "false",
            # Use PyTorch native SDPA instead of flash-attn package
            "ACESTEP_USE_FLASH_ATTENTION": "false",
            # GCS output
            "ACESTEP_GCS_BUCKET": GCS_BUCKET,
            "ACESTEP_GCS_PUBLIC_URL": f"https://storage.googleapis.com/{GCS_BUCKET}",
            # Auto-load LoRA adapter(s) on startup
            # Comma-separated, optional "name=path" syntax
            "ACESTEP_LORA_PATH": "alexayers=/app/loras/alexayers/final",
        }
    )
)


# ---------------------------------------------------------------------------
# Web endpoint
# ---------------------------------------------------------------------------
@app.cls(
    image=image,
    gpu=GPU,
    secrets=[GCS_SECRET, API_KEY_SECRET],
    timeout=600,
    scaledown_window=300,
)
@modal.concurrent(max_inputs=15)
class Server:
    @modal.enter()
    def setup_gcs_credentials(self):
        """Write the GCS service account JSON from the Modal secret to disk."""
        import os

        creds_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON", "")
        if creds_json:
            creds_path = "/tmp/gcs-key.json"
            with open(creds_path, "w") as f:
                f.write(creds_json)
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path

    @modal.asgi_app()
    def serve(self):
        import os

        from fastapi import Request
        from fastapi.responses import JSONResponse

        from acestep.api_server import app as fastapi_app

        @fastapi_app.middleware("http")
        async def verify_api_key(request: Request, call_next):
            if request.url.path == "/health":
                return await call_next(request)
            api_key = os.environ.get("MODAL_API_KEY")
            if api_key and request.headers.get("X-Modal-Api-Key") != api_key:
                return JSONResponse(
                    status_code=403, content={"detail": "Invalid API key"}
                )
            return await call_next(request)

        return fastapi_app
