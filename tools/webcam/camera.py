import os
import platform

import av
import cv2 as cv


def _is_uvc_device(camera_id) -> bool:
  """Allow only real USB (uvcvideo) devices on Linux.

  Opening tegra-video/VI nodes (CSI/GMSL raw capture) with OpenCV can stall the
  V4L2 subsystem at kernel level and freeze the whole board - never do it here.
  """
  if platform.system() != "Linux":
    return True
  dev = f"/dev/video{camera_id}" if isinstance(camera_id, int) else str(camera_id)
  name = os.path.basename(dev)
  link = f"/sys/class/video4linux/{name}/device/driver"
  try:
    driver = os.path.basename(os.readlink(link))
    return driver == "uvcvideo"
  except OSError:
    return False


class Camera:
  def __init__(self, cam_type_state, stream_type, camera_id):
    try:
      camera_id = int(camera_id)
    except ValueError: # allow strings, ex: /dev/video0
      pass
    self.cam_type_state = cam_type_state
    self.stream_type = stream_type
    self.cur_frame_id = 0

    print(f"Opening {cam_type_state} at {camera_id}")

    if not _is_uvc_device(camera_id):
      raise ValueError(
        f"{camera_id} is not a UVC (USB) device - refusing to open non-uvcvideo "
        "V4L2 nodes from the webcam path (kernel-stall risk)"
      )

    self.cap = cv.VideoCapture(camera_id)

    self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280.0)
    self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720.0)
    self.cap.set(cv.CAP_PROP_FPS, 25.0)

    # Optional MJPEG capture (USE_MJPEG=1): two UVC cameras on one USB controller
    # can exhaust uncompressed bandwidth (~55MB/s each at 720p YUYV) and drop frames.
    # MJPG keeps per-camera bandwidth ~10x lower; decode happens in cv.imdecode.
    self._mjpeg = os.getenv("USE_MJPEG") == "1"
    if self._mjpeg:
      self.cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))

    self.W = self.cap.get(cv.CAP_PROP_FRAME_WIDTH)
    self.H = self.cap.get(cv.CAP_PROP_FRAME_HEIGHT)

  @classmethod
  def bgr2nv12(self, bgr):
    frame = av.VideoFrame.from_ndarray(bgr, format='bgr24')
    return frame.reformat(format='nv12').to_ndarray()

  def read_frames(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if self._mjpeg:
        # MJPG frames come as full-color already via cap.read(); nothing extra needed,
        # kept explicit for future raw-JPEG handling (e.g. nvv4l2decoder offload).
        pass
      # Rotate the frame 180 degrees (flip both axes)
      frame = cv.flip(frame, -1)
      yuv = Camera.bgr2nv12(frame)
      yield yuv.data.tobytes()
    self.cap.release()
