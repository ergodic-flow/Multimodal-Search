import base64
import io
import click
import numpy as np
import torch
import torch.nn.functional as F
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from typing import List, Optional
from transformers import AutoModel, AutoImageProcessor, AutoTokenizer


class EmbedRequest(BaseModel):
    image: str
    model: str = "facebook/dino-vits16"
    mode: str = "image"
    bbox: Optional[dict] = None
    patch_grid: Optional[dict] = None


class EmbedResponse(BaseModel):
    vector: List[float]
    model: str
    dimension: int
    mode: str


class TextEmbedRequest(BaseModel):
    text: str
    model: str = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"


MODELS = {
    "dinov2_small": {
        "hf_id": "facebook/dinov2-small",
        "patch_size": 14,
        "image_size": 224,
    },
    "clip_vitb32_laion2b": {
        "hf_id": "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
        "patch_size": 32,
        "image_size": 224,
        "image_features": True,
        "text_enabled": True,
    },
}


class Embedder:
    def __init__(self):
        self.models = {}
        self.processors = {}
        self.tokenizers = {}

    def load(self, model_id: str):
        if model_id in self.models:
            return
        cfg = MODELS[model_id]
        hf_id = cfg["hf_id"]
        print(f"Loading {hf_id}...")
        self.processors[model_id] = AutoImageProcessor.from_pretrained(hf_id)
        if cfg.get("text_enabled"):
            self.tokenizers[model_id] = AutoTokenizer.from_pretrained(hf_id)
        self.models[model_id] = AutoModel.from_pretrained(hf_id)
        self.models[model_id].eval()
        print(f"Loaded {hf_id}")

    def embed(self, req: EmbedRequest) -> EmbedResponse:
        if req.model not in MODELS:
            raise HTTPException(400, f"Unknown model '{req.model}'. Available: {list(MODELS.keys())}")

        self.load(req.model)
        model = self.models[req.model]
        processor = self.processors[req.model]
        cfg = MODELS[req.model]

        img_bytes = base64.b64decode(req.image)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        if req.mode == "patch" and req.bbox:
            img = self._crop_bbox(img, req.bbox)

        inputs = processor(images=img, return_tensors="pt")

        with torch.no_grad():
            if cfg.get("image_features"):
                vector = model.get_image_features(**inputs).squeeze(0)
            else:
                outputs = model(**inputs)
                vector = self._embed_image(outputs)

        vector = vector.cpu().numpy().astype(np.float32)
        vector = (vector / np.linalg.norm(vector)).tolist()

        return EmbedResponse(
            vector=vector,
            model=req.model,
            dimension=len(vector),
            mode=req.mode,
        )

    def _embed_image(self, outputs) -> torch.Tensor:
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output.squeeze(0)
        return outputs.last_hidden_state[:, 0].squeeze(0)

    def _crop_bbox(self, img: Image.Image, bbox: dict) -> Image.Image:
        w, h = img.size
        left = bbox["x"] * w
        upper = bbox["y"] * h
        right = (bbox["x"] + bbox["w"]) * w
        lower = (bbox["y"] + bbox["h"]) * h
        return img.crop((left, upper, right, lower))

    def embed_text(self, req: TextEmbedRequest) -> EmbedResponse:
        if req.model not in MODELS:
            raise HTTPException(400, f"Unknown model '{req.model}'. Available: {list(MODELS.keys())}")
        cfg = MODELS[req.model]
        if not cfg.get("text_enabled"):
            raise HTTPException(400, f"Model '{req.model}' does not support text embedding")

        self.load(req.model)
        model = self.models[req.model]
        tokenizer = self.tokenizers[req.model]

        inputs = tokenizer(text=[req.text], return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            vector = model.get_text_features(**inputs).squeeze(0)

        vector = vector.cpu().numpy().astype(np.float32)
        vector = (vector / np.linalg.norm(vector)).tolist()

        return EmbedResponse(
            vector=vector,
            model=req.model,
            dimension=len(vector),
            mode="text",
        )


def create_app() -> FastAPI:
    embedder = Embedder()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.embedder = embedder
        yield

    app = FastAPI(title="Vision Embedder", version="0.1.0", lifespan=lifespan)

    @app.get("/models")
    def list_models():
        return {
            "models": [
                {
                    "id": mid,
                    "name": cfg["hf_id"],
                    "patch_size": cfg["patch_size"],
                    "image_size": cfg["image_size"],
                    "text_enabled": cfg.get("text_enabled", False),
                }
                for mid, cfg in MODELS.items()
            ]
        }

    @app.post("/embed", response_model=EmbedResponse)
    def embed(req: EmbedRequest):
        return embedder.embed(req)

    @app.post("/embed_text", response_model=EmbedResponse)
    def embed_text(req: TextEmbedRequest):
        return embedder.embed_text(req)

    return app


@click.command()
@click.option("--host", default="127.0.0.1")
@click.option("--port", default=8001, type=int)
def main(host: str, port: int):
    app = create_app()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
