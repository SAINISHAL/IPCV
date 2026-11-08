"""OpenCV capture worker for the dashboard's live video feed."""
from collections import deque
from dataclasses import dataclass
from sys import platform
from threading import Event, Lock, Thread
from time import monotonic
from typing import Callable

import cv2

from src.utils.logger import setup_logger

logger = setup_logger("LiveCamera")


def open_webcam(device_id: int, settings: dict) -> cv2.VideoCapture:
    """Open a local camera and release failed backend attempts."""
    backends = (cv2.CAP_DSHOW, cv2.CAP_ANY) if platform == "win32" else (cv2.CAP_ANY,)
    for backend in backends:
        try:
            capture = cv2.VideoCapture(device_id, backend)
        except cv2.error:
            continue
        if not capture.isOpened():
            capture.release()
            continue
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, settings["frame_width"])
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, settings["frame_height"])
        capture.set(cv2.CAP_PROP_FPS, settings["target_fps"])
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    raise RuntimeError(f"Could not open camera {device_id}. Check the system camera permissions, "
                       "close other apps using the camera, or try a different camera device.")


@dataclass(frozen=True)
class CameraFrame:
    jpeg: bytes
    detected: int
    recognized: int
    unknown: int
    faces: tuple[tuple[int, str], ...]
    fps: float | None


class LiveCamera:
    """Own one capture and publish the latest complete result without queuing frames."""

    def __init__(self, device_id: int, settings: dict, process_frame: Callable):
        self.device_id = device_id
        self.settings = settings
        self.process_frame = process_frame
        self._lock = Lock()
        self._stop = Event()
        self._last_poll = monotonic()
        self._status = "Starting camera…"
        self._frame = None
        self._thread = Thread(target=self._run, name=f"webcam-{device_id}", daemon=True)
        self._thread.start()

    @property
    def active(self) -> bool:
        return self._thread.is_alive()

    def stop(self) -> None:
        self._stop.set()

    def snapshot(self) -> tuple[str, CameraFrame | None]:
        with self._lock:
            self._last_poll = monotonic()
            if self.active and self._stop.is_set():
                return "Stopping camera…", None
            return self._status, self._frame

    def _run(self) -> None:
        capture = None
        status = "Camera stopped."
        times = deque(maxlen=20)
        try:
            capture = open_webcam(self.device_id, self.settings)
            consecutive_failures = 0
            while not self._stop.is_set():
                # Release the camera even if the browser tab was closed without pressing Stop.
                with self._lock:
                    if monotonic() - self._last_poll > 20:
                        break
                ok, frame = capture.read()
                if not ok or frame is None or frame.size == 0:
                    consecutive_failures += 1
                    if consecutive_failures >= 20:
                        raise RuntimeError("The camera stopped sending frames. Check camera permissions "
                                           "and the connection, then try Start Camera again.")
                    self._stop.wait(0.1)
                    continue
                consecutive_failures = 0
                annotated, meta = self.process_frame(frame)
                ok, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ok:
                    raise RuntimeError("Could not display the camera frame. Try restarting the camera.")
                times.append(monotonic())
                fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else None
                faces = tuple((face["track_id"], face["identity"]) for face in meta["faces"])
                with self._lock:
                    self._frame = CameraFrame(jpeg.tobytes(), meta["detected"], meta["recognized"],
                                              meta["unknown"], faces, fps)
                    self._status = "Live"
        except RuntimeError as exc:
            status = str(exc)
        except Exception:
            logger.exception("Live camera processing failed.")
            status = "Camera processing failed. Check the connection and try Start Camera again."
        finally:
            if capture is not None:
                capture.release()
            with self._lock:
                self._frame = None
                self._status = status
