# Cortex: one platform, about 48 vision models

## What it was
Cortex was an internal platform at UNADA Labs where we brought around 48 computer vision models together behind one system. The idea was simple: different use cases need different objects detected, and no single model is good at all of them. So instead of one huge model, we had many specialised ones and Cortex decided which ones to run.

The models were a mix of:
- **Our own trained detectors**, fine-tuned on labelled data for specific objects
- **Vision Transformers (ViT)**, which split an image into patches and use attention to understand the whole scene
- **GAN-based models**, used on the generation and enhancement side

## How a request flowed through it
1. An image or a video comes in along with the use case.
2. If it's a video, we sample frames instead of running every frame. At 30 fps, two frames a few milliseconds apart are basically the same picture, so running both just burns compute.
3. Each frame is preprocessed into the exact format the model was trained on.
4. Cortex looks up which models are registered for that use case and runs them.
5. The raw detections get cleaned up: low confidence boxes are dropped and overlapping duplicates are removed with non-max suppression.
6. Clean results go back as label, box and confidence.

## What's in this folder
The model weights belong to the company, so `model_router.py` uses dummy models. Everything around them is real logic: the model registry, routing by use case, frame sampling and the post-processing (confidence filter and NMS).

```bash
python model_router.py
```

## What I took away from it
Most of the work in a multi-model system isn't the models, it's everything around them: keeping track of which model does what, making outputs from different models look the same, and not wasting GPU time on frames that add nothing.
