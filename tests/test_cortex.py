import pytest

from model_router import Detection, demo_registry, postprocess, run, sample_frames


def test_nms_removes_duplicates_and_weak_boxes():
    dets = [Detection("helmet", (10, 10, 50, 50), 0.9), Detection("helmet", (11, 11, 51, 51), 0.8),
            Detection("helmet", (100, 100, 140, 140), 0.2)]
    assert len(postprocess(dets)) == 1


def test_frame_sampling():
    assert len(sample_frames(300, fps=30, every_seconds=1)) == 10


def test_routing_by_use_case():
    reg = demo_registry()
    assert run(reg, "traffic", [0])[0][0].label == "truck"
    with pytest.raises(KeyError):
        run(reg, "unknown", [0])
