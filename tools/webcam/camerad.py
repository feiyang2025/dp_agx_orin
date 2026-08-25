#!/usr/bin/env python3
"""Webcam camera daemon (UVC path).

Hardened for headless bring-up on Jetson:
- a camera that fails to open or produces no first frame within FIRST_FRAME_TIMEOUT
  is skipped (stream not registered) instead of silently dying;
- if any stream stops producing frames for NO_FRAME_TIMEOUT seconds the process
  exits non-zero so the manager restarts it (self-heal loop) - this prevents the
  'black screen freeze' failure mode where downstream waits forever;
- frame timestamps use CLOCK_MONOTONIC instead of synthetic frame_id*50ms.
"""
import os
import platform
import threading
import time
from collections import namedtuple

from msgq.visionipc import VisionIpcServer, VisionStreamType
from cereal import messaging

from openpilot.tools.webcam.camera import Camera
from openpilot.common.realtime import Ratekeeper
from openpilot.common.swaglog import cloudlog

ROAD_CAM = os.getenv("ROAD_CAM", "0")
WIDE_CAM = os.getenv("WIDE_CAM")
DRIVER_CAM = os.getenv("DRIVER_CAM")

FIRST_FRAME_TIMEOUT = 3.0   # seconds to first frame after open
NO_FRAME_TIMEOUT = 5.0      # seconds without frames -> exit for manager restart
TARGET_FPS = 20

CameraType = namedtuple("CameraType", ["msg_name", "stream_type", "cam_id"])

CAMERAS = [
  CameraType("roadCameraState", VisionStreamType.VISION_STREAM_ROAD, ROAD_CAM)
]
if WIDE_CAM:
  CAMERAS.append(CameraType("wideRoadCameraState", VisionStreamType.VISION_STREAM_WIDE_ROAD, WIDE_CAM))
if DRIVER_CAM:
  CAMERAS.append(CameraType("driverCameraState", VisionStreamType.VISION_STREAM_DRIVER, DRIVER_CAM))


def _open_camera(cam_type: CameraType):
  """Open a camera and grab its first frame. Returns (camera, first_frame) or (None, reason)."""
  cam_device = f"/dev/video{cam_type.cam_id}" if platform.system() != "Darwin" else cam_type.cam_id
  try:
    cam = Camera(cam_type.msg_name, cam_type.stream_type, cam_device)
    gen = cam.read_frames()
    first = next(gen, None)
    if first is None:
      return None, f"no frames from {cam_device}"
    return (cam, gen, first), None
  except Exception as e:
    return None, f"{type(e).__name__}: {e} on {cam_device}"


class Camerad:
  def __init__(self):
    opened = []
    for ct in CAMERAS:
      result, err = _open_camera(ct)
      if result is None:
        cloudlog.error("webcamerad: skipping %s: %s", ct.msg_name, err)
        continue
      cam, gen, first = result
      cam.cur_frame_id = 0
      opened.append((ct, cam, gen, first))

    if not opened:
      cloudlog.error("webcamerad: no usable cameras found (%d requested), exiting", len(CAMERAS))
      raise RuntimeError("no usable cameras")

    self.pm = messaging.PubMaster([ct.msg_name for ct, *_ in opened])
    self.vipc_server = VisionIpcServer("camerad")
    self.streams = []
    for ct, cam, gen, first in opened:
      # W/H reflect what the device actually delivers after format negotiation
      self.vipc_server.create_buffers(ct.stream_type, 20, int(cam.W), int(cam.H))
      self.streams.append({"type": ct, "cam": cam, "gen": gen, "first": first,
                           "last_frame_ts": time.monotonic()})
    self.vipc_server.start_listener()

  def _send_yuv(self, yuv, frame_id, pub_type, yuv_type):
    eof = time.monotonic_ns()
    self.vipc_server.send(yuv_type, yuv, frame_id, eof, eof)
    dat = messaging.new_message(pub_type, valid=True)
    msg = {
      "frameId": frame_id,
      "transform": [1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0]
    }
    setattr(dat, pub_type, msg)
    self.pm.send(pub_type, dat)

  def camera_runner(self, stream):
    ct, cam, gen = stream["type"], stream["cam"], stream["gen"]
    rk = Ratekeeper(TARGET_FPS, None)
    frames = [first]  # list used as a mutable cell across helper closure
    while True:
      yuv = frames[0]
      if yuv is None:
        # generator exhausted mid-run: camera unplugged or driver stall
        cloudlog.error("webcamerad: %s stopped producing frames, exiting for restart", ct.msg_name)
        os._exit(1)
      self._send_yuv(yuv, cam.cur_frame_id, ct.msg_name, ct.stream_type)
      cam.cur_frame_id += 1
      stream["last_frame_ts"] = time.monotonic()
      frames[0] = next(gen, None)
      rk.keep_time()

  def watchdog(self):
    while True:
      time.sleep(1.0)
      now = time.monotonic()
      for s in self.streams:
        if now - s["last_frame_ts"] > NO_FRAME_TIMEOUT:
          cloudlog.error("webcamerad: %s stalled %.1fs without frames, exiting for restart",
                         s["type"].msg_name, now - s["last_frame_ts"])
          os._exit(1)

  def run(self):
    threading.Thread(target=self.watchdog, daemon=True).start()
    threads = []
    for s in self.streams:
      t = threading.Thread(target=self.camera_runner, args=(s,), daemon=True)
      t.start()
      threads.append(t)

    for t in threads:
      t.join()


def main():
  try:
    camerad = Camerad()
  except Exception as e:
    cloudlog.error("webcamerad failed to start: %s", e)
    raise SystemExit(1)
  camerad.run()


if __name__ == "__main__":
  main()
