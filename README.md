# Multimodal Search

Embedding-based image and text retrieval for a folder of images.
Supports image/text/patch search. Useful for exploring and viewing your images.

## Getting Started

Move your folder of images to `./images/`. Run `scripts/clip_embed.py` and `scripts/dino_embed.py` on the images to produce vectors.

Launch the app with `docker compose up`. Open `http://localhost:3000` to see the UI.

## Screenshots

### Image Search
![image search](/assets/image.jpeg)

### Patch Search
![patch search](/assets/patch.jpeg)

### Text Search
![text search](/assets/text.jpeg)
