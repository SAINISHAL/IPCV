from threading import Event
from time import monotonic, sleep

import numpy as np

from src.webcam import live_camera


def wait_for(camera, predicate):
    deadline = monotonic() + 3
    while monotonic() < deadline:
        result = camera.snapshot()
        if predicate(result):
            return result
        sleep(0.01)
    raise AssertionError(f"Camera did not reach the expected state: {camera.snapshot()[0]}")


def test_camera_publishes_video_and_releases_the_selected_device(monkeypatch):
    released = Event()

    class Capture:
        def read(self):
            sleep(0.01)
            return True, np.zeros((48, 64, 3), dtype=np.uint8)

        def release(self):
            released.set()

    def open_camera(device_id, settings):
        assert device_id == 2
        return Capture()

    def process(frame):
        return frame, {"detected": 2, "recognized": 1, "unknown": 1,
                       "faces": [{"track_id": 0, "identity": "Student A"},
                                 {"track_id": 1, "identity": "UNKNOWN"}]}

    monkeypatch.setattr(live_camera, "open_webcam", open_camera)
    camera = live_camera.LiveCamera(2, {}, process)
    try:
        _, frame = wait_for(camera, lambda result: result[1] is not None and result[1].fps is not None)
        assert frame.jpeg.startswith(b"\xff\xd8")
        assert (frame.detected, frame.recognized, frame.unknown) == (2, 1, 1)
        assert frame.faces == ((0, "Student A"), (1, "UNKNOWN"))
        assert frame.fps > 0
    finally:
        camera.stop()
    assert released.wait(2)
    wait_for(camera, lambda result: result == ("Camera stopped.", None))


def test_camera_permission_error_is_reported_without_a_frame(monkeypatch):
    def denied(device_id, settings):
        raise RuntimeError("Could not open camera. Check camera permissions.")

    monkeypatch.setattr(live_camera, "open_webcam", denied)
    camera = live_camera.LiveCamera(0, {}, lambda frame: None)
    status, frame = wait_for(camera, lambda _: not camera.active)
    assert "camera permissions" in status
    assert frame is None
