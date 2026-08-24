"""Unified camera source abstraction for DragonPilot on Jetson.

All backends produce NV12 frames consumed by VisionIPC, regardless of the
physical interface:

  UvcBackend   USB UVC cameras (road/wide), OpenCV capture            [Phase 1]
  GmslBackend  onboard GMSL2 deserializer -> Argus ISP -> NV12        [Phase 2]

Downstream (modeld / UI) requires no changes: backends only replace the
capture+convert stage of tools/webcam/camerad.py.
"""
from abc import ABC, abstractmethod

import numpy as np

from openpilot.tools.webcam.camera import Camera


class CameraSource(ABC):
  """A single camera producing NV12 frames at a fixed resolution/fps."""

  def __init__(self, msg_name: str, stream_type, width: int = 1280, height: int = 720, fps: int = 20):
    self.msg_name = msg_name
    self.stream_type = stream_type
    self.width = width
    self.height = height
    self.fps = fps
    self.cur_frame_id = 0

  @abstractmethod
  def open(self) -> bool:
    """Initialize the device. Returns False if unavailable."""

  @abstractmethod
  def read_frame(self) -> np.ndarray | None:
    """Return one NV12 frame (uint8, H*3/2 x W) or None if not ready."""

  @abstractmethod
  def close(self):
    pass


class UvcBackend(CameraSource):
  """USB UVC camera via OpenCV (existing path)."""

  def __init__(self, msg_name: str, stream_type, camera_id,
               width: int = 1280, height: int = 720, fps: int = 20):
    super().__init__(msg_name, stream_type, width, height, fps)
    self._camera_id = camera_id
    self._cam: Camera | None = None
    self._gen = None

  def open(self) -> bool:
    try:
      self._cam = Camera(self.msg_name, self.stream_type, self._camera_id)
      first = next(self._cam.read_frames(), None)
      if first is None:
        return False
      self._gen = self._iter_frames(first)
      self.width, self.height = self._cam.W, self._cam.H
      return True
    except Exception:
      self._cam = None
      return False

  def _iter_frames(self, first):
    yield first
    yield from self._cam.read_frames()

  def read_frame(self) -> np.ndarray | None:
    return next(self._gen, None)

  def close(self):
    if self._cam is not None:
      try:
        self._cam.cap.release()
      except Exception:
        pass
      self._cam = None
      self._gen = None


class GmslBackend(CameraSource):
  """Onboard GMSL2 camera (e.g. IMX390) via nvarguscamerasrc / Argus ISP.

  Phase 2: requires vendor BSP (device tree + deserializer/sensor driver +
  Argus override). ISP outputs NV12 directly; no software conversion needed.
  """

  def __init__(self, msg_name: str, stream_type, sensor_id: int,
               width: int = 1920, height: int = 1080, fps: int = 20):
    super().__init__(msg_name, stream_type, width, height, fps)
    self._sensor_id = sensor_id

  def open(self) -> bool:
    raise NotImplementedError("GMSL support lands in Phase 2 (vendor BSP required)")

  def read_frame(self) -> np.ndarray | None:
    raise NotImplementedError

  def close(self):
    pass
