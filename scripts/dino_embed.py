import json
import click
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoImageProcessor

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
HF_ID = "facebook/dinov2-small"


@click.command()
@click.option("--image-dir", type=click.Path(exists=True, file_okay=False), required=True)
@click.option("--output", type=click.Path(), default="data/vectors_dinov2_small.jsonl", show_default=True)
def main(image_dir: str, output: str):
    click.echo(f"Loading {HF_ID}...")
    processor = AutoImageProcessor.from_pretrained(HF_ID)
    model = AutoModel.from_pretrained(HF_ID)
    model.eval()

    images = sorted(p for p in Path(image_dir).rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        click.secho(f"No images found in {image_dir}", fg="red")
        raise SystemExit(1)

    click.echo(f"Embedding {len(images)} images...")
    with open(output, "w") as f:
        for img_path in tqdm(images, desc="Embedding"):
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                click.secho(f"  Skipping {img_path}: {e}", fg="yellow")
                continue

            inputs = processor(images=img, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)

            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                vector = outputs.pooler_output.squeeze(0)
            else:
                vector = outputs.last_hidden_state[:, 0].squeeze(0)

            vector = vector.cpu().numpy().astype(np.float32)
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm

            record = {"id": img_path.name, "v": vector.tolist()}
            f.write(json.dumps(record) + "\n")

    click.secho(f"Done. Wrote {output}", fg="green")


if __name__ == "__main__":
    main()
