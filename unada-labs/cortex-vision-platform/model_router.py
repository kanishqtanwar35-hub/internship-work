"""How Cortex routed requests to the right vision model.

The real platform had around 48 models (our own trained detectors, ViTs, GANs).
The weights belong to the company, so the models below are dummies, but the
routing, video frame sampling and post-processing work the same way.
"""
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Detection:
    label: str
    box: tuple  # (x1, y1, x2, y2)
    confidence: float


@dataclass
class ModelEntry:
    name: str
    family: str          # "detector", "vit" or "gan"
    use_cases: set
    predict: Callable


@dataclass
class Registry:
    models: dict = field(default_factory=dict)

    def register(self, entry):
        self.models[entry.name] = entry

    def for_use_case(self, use_case):
        found = [m for m in self.models.values() if use_case in m.use_cases]
        if not found:
            raise KeyError(f"no model registered for use case '{use_case}'")
        return found


def iou(a, b):
    """Overlap between two boxes, 0 = no overlap, 1 = same box."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)

    def area(r):
        return (r[2] - r[0]) * (r[3] - r[1])

    union = area(a) + area(b) - inter
    return inter / union if union else 0.0


def postprocess(dets, min_conf=0.5, iou_thresh=0.5):
    """Drop weak detections, then non-max suppression to remove duplicate boxes."""
    dets = sorted([d for d in dets if d.confidence >= min_conf], key=lambda d: -d.confidence)
    kept = []
    for d in dets:
        if all(d.label != k.label or iou(d.box, k.box) < iou_thresh for k in kept):
            kept.append(d)
    return kept


def sample_frames(total_frames, fps, every_seconds=1.0):
    """For video we don't run every frame, consecutive frames are almost identical."""
    step = max(1, int(round(fps * every_seconds)))
    return list(range(0, total_frames, step))


def run(registry, use_case, frames):
    results = []
    for f in frames:
        dets = []
        for model in registry.for_use_case(use_case):
            dets.extend(model.predict(f))
        results.append(postprocess(dets))
    return results


def demo_registry():
    reg = Registry()
    reg.register(ModelEntry(
        "helmet-detector-v3", "detector", {"site-safety"},
        lambda f: [Detection("helmet", (10, 10, 50, 50), 0.91),
                   Detection("helmet", (12, 11, 51, 52), 0.80),     # duplicate box
                   Detection("helmet", (200, 40, 240, 80), 0.30)]))  # too weak
    reg.register(ModelEntry(
        "vehicle-vit", "vit", {"traffic"},
        lambda f: [Detection("truck", (0, 0, 100, 60), 0.88)]))
    return reg


if __name__ == "__main__":
    reg = demo_registry()
    frames = sample_frames(total_frames=300, fps=30)  # 10 second clip -> 10 frames
    out = run(reg, "site-safety", frames)
    print(f"{len(frames)} frames sampled out of 300")
    print("frame 0 after cleanup:", out[0])
