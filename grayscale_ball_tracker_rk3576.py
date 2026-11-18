"""RK3576 / K7C grayscale steel-ball tracker with LAN MJPEG streaming.

Ported from the K230 grayscale_ball_tracker.py logic.
This version uses Linux + OpenCV + V4L2, and keeps the heavy work inside the
cropped ROI so the board can spend more time on capture and streaming.
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import math
import os
import platform
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import cv2
import numpy as np

try:
    cv2.setUseOptimized(True)
    cv2.setNumThreads(1)
except Exception:
    pass

try:
    import serial
except ImportError:  # pragma: no cover - optional dependency
    serial = None


# Base calibration copied from the original K230 script.
BASE_FRAME_WIDTH = 640
BASE_FRAME_HEIGHT = 480
BALL_DIAMETER_MM = 10.0
BALL_DIAMETER_PIXELS_BASE = 20.0

# Original ROI from the K230 version, expressed in the 640x480 frame.
ROI_BASE_X = 20
ROI_BASE_Y = 218
ROI_BASE_W = 600
ROI_BASE_H = 55

# Camera defaults for the RK3576 board.
DEFAULT_CAMERA_INDEX = 73
CAMERA_CANDIDATES = (73, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
REQUESTED_WIDTH = 640
REQUESTED_HEIGHT = 480
REQUESTED_FPS = 30

# LAN MJPEG stream and MP4 recording.
STREAM_PORT = 8081
ROI_STREAM_WIDTH = 1024
STREAM_JPEG_QUALITY = 80
STREAM_TARGET_FPS = 20
RECORDING_FPS = 20.0
RECORDING_FOURCC = "mp4v"
RECORDING_DIRECTORY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "recordings"
)

# Grayscale detector.
HOUGH_DP = 1.0
HOUGH_MIN_DISTANCE = 18
HOUGH_CANNY_THRESHOLD = 60
HOUGH_SCORE_THRESHOLD = 12
MIN_RADIUS = 4
MAX_RADIUS = 30
GAUSSIAN_KERNEL = 3
GAUSSIAN_SIGMA = 0.8
ENABLE_HOUGH_REFINEMENT = False
HOUGH_EVERY_N_FRAMES = 5

BLACKHAT_KERNEL_SIZE = 21
BLACKHAT_BLUR_KERNEL = 5
MIN_BLACKHAT_RESPONSE = 12.0
MIN_TRACKED_RESPONSE = 7.0
FALLBACK_RADIUS = 10
BLACKHAT_CENTER_Y_OFFSET = -5

SEARCH_X_MARGIN = 0
SEARCH_X_MIN = 95
SEARCH_X_MAX = 505
SEARCH_Y_MIN = 12
SEARCH_Y_MAX = 55
CENTERLINE_HALF_HEIGHT = 2
CENTERLINE_PIXEL_GAP = 2
CENTERLINE_SMOOTH_KERNEL = 5
CENTERLINE_OUTSIDE_BAND = 5
CENTERLINE_EDGE_FLOOR = 4.0
CENTERLINE_NOISE_MULTIPLIER = 3.0
CENTERLINE_MIN_DARK_CONTRAST = 5.0
CENTERLINE_MIN_DARK_RUN_RATIO = 0.65
CENTERLINE_MIN_EDGE_BALANCE = 0.28
MAX_ANCHOR_DISTANCE = 22
TRACK_SEARCH_X = 100
TRACK_SEARCH_Y = 28

MIN_LOCAL_CONTRAST = 2.0
MIN_CANDIDATE_CONTRAST = 1.5
MAX_CIRCLES = 20
EDGE_MARGIN = 3

# Temporal bright-reference detector. Stationary tube edges cancel out, while
# the moving steel ball remains a compact darker-than-background region.
BACKGROUND_WARMUP_FRAMES = 15
BACKGROUND_UPDATE_EVERY_N_FRAMES = 3
BACKGROUND_DECAY = 1
MIN_BACKGROUND_RESPONSE = 10.0
MIN_TRACKED_BACKGROUND_RESPONSE = 6.0
BACKGROUND_THRESHOLD_RATIO = 0.42
MIN_BACKGROUND_AREA = 35
MAX_BACKGROUND_AREA = 520
MIN_BACKGROUND_SIZE = 4
MAX_BACKGROUND_WIDTH = 30
MAX_BACKGROUND_HEIGHT = 30
MIN_BALL_EQUIVALENT_DIAMETER = 9
MAX_BALL_EQUIVALENT_DIAMETER = 27
MIN_BACKGROUND_ASPECT = 0.30
MAX_BACKGROUND_ASPECT = 3.20
MIN_BACKGROUND_FILL = 0.12
MIN_CONTOUR_CIRCULARITY = 0.08
MIN_HULL_CIRCULARITY = 0.34
MIN_CIRCLE_FILL = 0.20
MIN_SOLIDITY = 0.38
MIN_TEXTURE_STD = 3.0
HIGH_CONTRAST_KERNEL_X = 31
HIGH_CONTRAST_KERNEL_Y = 17
MIN_HIGH_CONTRAST_PEAK = 14.0
MIN_HIGH_CONTRAST_MEAN = 4.0
BACKGROUND_BLUR_KERNEL = 3
BACKGROUND_OPEN_KERNEL = 3
BACKGROUND_CLOSE_KERNEL = 7
ILLUMINATION_KERNEL_X = 61
ILLUMINATION_KERNEL_Y = 31
PREVIEW_CONTRAST = 1.35
PREVIEW_OFFSET = -45.0
SATURATION_THRESHOLD = 250
SATURATION_SAMPLE_EVERY = 15
BACKGROUND_FREEZE_MARGIN = 10
BALL_PATH_TOLERANCE = 22
TRACK_PATH_TOLERANCE = 28

CONFIRM_FRAMES = 3
LOST_HOLD_FRAMES = 4
MAX_CENTER_JUMP = 100
POSITION_ALPHA = 0.70
FAST_POSITION_ALPHA = 0.92
FAST_MOTION_THRESHOLD = 6.0
RADIUS_ALPHA = 0.35
VELOCITY_ALPHA = 0.65
VELOCITY_DECAY = 0.75
DISPLAY_PREDICTION = 0.35
BOX_MARGIN = 3
AXIS_COLOR = (0, 255, 0)
TEXT_COLOR = (255, 255, 255)
BOX_COLOR = (0, 0, 255)

# Optional UART output.
UART_ENABLED = False
UART_PORT = "/dev/ttyS8"
UART_BAUD = 115200


def scale_roi(frame_width, frame_height, base_y=ROI_BASE_Y, base_h=ROI_BASE_H):
    sx = float(frame_width) / float(BASE_FRAME_WIDTH)
    sy = float(frame_height) / float(BASE_FRAME_HEIGHT)
    roi_x = int(ROI_BASE_X * sx + 0.5)
    roi_y = int(base_y * sy + 0.5)
    roi_w = max(1, int(ROI_BASE_W * sx + 0.5))
    roi_h = max(1, int(base_h * sy + 0.5))
    return roi_x, roi_y, roi_w, roi_h


def scaled_int(value, scale, minimum=1):
    return max(minimum, int(value * scale + 0.5))


def scaled_odd(value, scale, minimum=3):
    result = scaled_int(value, scale, minimum)
    if result % 2 == 0:
        result += 1
    return result


def find_video_devices():
    if platform.system() != "Linux":
        return [(index, "camera-%d" % index) for index in CAMERA_CANDIDATES]

    devices = []
    def video_index(path):
        try:
            return int(os.path.basename(path).replace("video", ""))
        except ValueError:
            return 1000000

    for path in sorted(glob.glob("/sys/class/video4linux/video*"), key=video_index):
        try:
            index = int(os.path.basename(path).replace("video", ""))
        except ValueError:
            continue

        name_path = os.path.join(path, "name")
        name = "video%d" % index
        if os.path.exists(name_path):
            try:
                with open(name_path, "r", encoding="utf-8") as handle:
                    name = handle.read().strip()
            except Exception:
                pass

        lowered = name.lower()
        if any(token in lowered for token in ("metadata", "association", "statistics", "params", "meta")):
            continue

        devices.append((index, name))

    if devices:
        return devices
    return [(index, "video-%d" % index) for index in CAMERA_CANDIDATES]


def open_camera(
    camera_index,
    width,
    height,
    fps,
    exposure=None,
    gain=None,
    brightness=None,
):
    candidates = []
    if camera_index is None:
        candidates.extend(find_video_devices())
    else:
        candidates.append((camera_index, "requested"))
        for index, name in find_video_devices():
            if index != camera_index:
                candidates.append((index, name))

    for index, name in candidates:
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        controls = []
        if exposure is not None:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0)
            controls.append(("exposure", cv2.CAP_PROP_EXPOSURE, exposure))
        if gain is not None:
            controls.append(("gain", cv2.CAP_PROP_GAIN, gain))
        if brightness is not None:
            controls.append(("brightness", cv2.CAP_PROP_BRIGHTNESS, brightness))
        for label, property_id, requested_value in controls:
            accepted = cap.set(property_id, float(requested_value))
            actual_value = cap.get(property_id)
            print(
                "[camera] %s requested=%.2f actual=%.2f accepted=%s"
                % (label, requested_value, actual_value, accepted)
            )

        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            continue

        actual_height, actual_width = frame.shape[:2]
        actual_fps = float(cap.get(cv2.CAP_PROP_FPS))
        fourcc_value = int(cap.get(cv2.CAP_PROP_FOURCC))
        actual_fourcc = "".join(
            chr((fourcc_value >> (8 * offset)) & 0xFF)
            for offset in range(4)
        ).rstrip("\x00")
        print("[camera] opened /dev/video%d (%s)" % (index, name))
        gray_frame = (
            frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        )
        _, saturated_mask = cv2.threshold(
            gray_frame, SATURATION_THRESHOLD - 1, 255, cv2.THRESH_BINARY
        )
        saturation_percent = (
            100.0 * cv2.countNonZero(saturated_mask) / float(gray_frame.size)
        )
        print(
            "[camera] requested=%dx%d@%d actual=%dx%d@%.1f fourcc=%s"
            % (
                width, height, fps, actual_width, actual_height,
                actual_fps, actual_fourcc or "unknown",
            )
        )
        print("[camera] saturation=%.1f%%" % saturation_percent)
        if saturation_percent >= 35.0:
            print(
                "[camera] warning: image is overexposed; lower exposure/gain "
                "until SAT is below about 20%"
            )
        return cap, index, actual_width, actual_height

    raise RuntimeError("No usable V4L2 camera found")


class FrameGrabber:
    def __init__(self, cap):
        self.cap = cap
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.latest_frame = None
        self.latest_seq = 0
        self.running = False
        self.thread = None
        self.last_error = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        try:
            while self.running:
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    time.sleep(0.001)
                    continue
                with self.condition:
                    self.latest_frame = frame
                    self.latest_seq += 1
                    self.condition.notify_all()
        except BaseException as error:
            self.last_error = error
            self.running = False
            with self.condition:
                self.condition.notify_all()

    def wait_latest(self, last_seq, timeout=0.2):
        with self.condition:
            if self.running and self.latest_seq == last_seq:
                self.condition.wait(timeout)
            return self.latest_frame, self.latest_seq

    def stop(self):
        self.running = False
        with self.condition:
            self.condition.notify_all()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)


class MJPEGHandler(BaseHTTPRequestHandler):
    streamer = None

    def log_message(self, format, *args):
        return

    def do_GET(self):
        request_path = self.path.split("?", 1)[0]
        if request_path == "/":
            self._serve_index()
            return
        if request_path.startswith("/stream/"):
            self._serve_stream()
            return
        if request_path == "/api/record/status":
            self._send_json(self.streamer.recording_status())
            return
        if request_path == "/api/record/download":
            self._serve_recording()
            return
        self.send_error(404)

    def do_POST(self):
        request_path = self.path.split("?", 1)[0]
        if request_path == "/api/record/start":
            ok, message = self.streamer.start_recording()
            self._send_json(
                {"ok": ok, "message": message, **self.streamer.recording_status()},
                200 if ok else 409,
            )
            return
        if request_path == "/api/record/stop":
            ok, message = self.streamer.stop_recording()
            self._send_json(
                {"ok": ok, "message": message, **self.streamer.recording_status()},
                200 if ok else 409,
            )
            return
        self.send_error(404)

    def _send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_index(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        html = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>K7C Ball Tracker</title>
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; background: #111827; color: #e5e7eb; font-family: Arial, sans-serif; }
    .wrap { max-width: 1100px; margin: 0 auto; padding: 16px; }
    header { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 12px; }
    h1 { font-size: 18px; margin: 0; }
    .status { display: flex; align-items: center; gap: 7px; color: #9ca3af; font-size: 13px; }
    .dot { width: 9px; height: 9px; border-radius: 50%; background: #6b7280; }
    .dot.live { background: #ef4444; box-shadow: 0 0 0 3px rgba(239,68,68,.18); }
    .viewer { border: 1px solid #374151; background: #000; }
    img { width: 100%; height: auto; display: block; image-rendering: auto; }
    .toolbar { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; padding-top: 12px; }
    button, .download { min-height: 38px; border: 1px solid #4b5563; border-radius: 6px; padding: 0 14px; background: #1f2937; color: #f9fafb; font-size: 14px; cursor: pointer; text-decoration: none; display: inline-flex; align-items: center; }
    button.primary { background: #b91c1c; border-color: #dc2626; }
    button:disabled, .download.disabled { opacity: .42; cursor: default; pointer-events: none; }
    .meta { margin-left: auto; color: #9ca3af; font-size: 12px; }
    .error { min-height: 18px; padding-top: 8px; color: #fca5a5; font-size: 12px; }
    @media (max-width: 620px) { .meta { width: 100%; margin-left: 0; } }
  </style>
</head>
<body>
  <main class="wrap">
    <header>
      <h1>K7C Ball Tracker</h1>
      <div class="status"><span id="recordDot" class="dot"></span><span id="recordState">Ready</span></div>
    </header>
    <section class="viewer"><img src="/stream/0" alt="Ball tracker stream"></section>
    <div class="toolbar">
      <button id="startButton" class="primary" type="button">Record</button>
      <button id="stopButton" type="button" disabled>Stop</button>
      <a id="downloadButton" class="download disabled" href="/api/record/download">Download MP4</a>
      <span id="recordMeta" class="meta">0 frames</span>
    </div>
    <div id="recordError" class="error"></div>
  </main>
  <script>
    const startButton = document.getElementById('startButton');
    const stopButton = document.getElementById('stopButton');
    const downloadButton = document.getElementById('downloadButton');
    const recordDot = document.getElementById('recordDot');
    const recordState = document.getElementById('recordState');
    const recordMeta = document.getElementById('recordMeta');
    const recordError = document.getElementById('recordError');

    function renderStatus(data) {
      const active = Boolean(data.recording);
      recordDot.classList.toggle('live', active);
      recordState.textContent = active ? 'Recording' : (data.ready ? 'Saved' : 'Ready');
      startButton.disabled = active;
      stopButton.disabled = !active;
      downloadButton.classList.toggle('disabled', !data.ready);
      const seconds = Number(data.seconds || 0).toFixed(1);
      recordMeta.textContent = (data.frames || 0) + ' frames / ' + seconds + ' s';
      recordError.textContent = data.error || '';
    }

    async function refreshStatus() {
      try {
        const response = await fetch('/api/record/status', {cache: 'no-store'});
        renderStatus(await response.json());
      } catch (error) {
        recordError.textContent = 'Recorder status unavailable';
      }
    }

    async function recordAction(endpoint) {
      recordError.textContent = '';
      try {
        const response = await fetch(endpoint, {method: 'POST'});
        const data = await response.json();
        renderStatus(data);
        if (!response.ok) recordError.textContent = data.message || 'Recorder error';
      } catch (error) {
        recordError.textContent = 'Recorder request failed';
      }
    }

    startButton.addEventListener('click', () => recordAction('/api/record/start'));
    stopButton.addEventListener('click', () => recordAction('/api/record/stop'));
    setInterval(refreshStatus, 1000);
    refreshStatus();
  </script>
</body>
</html>"""
        self.wfile.write(html.encode("utf-8"))

    def _serve_stream(self):
        try:
            stream_id = int(self.path.split("?", 1)[0].rsplit("/", 1)[-1])
        except ValueError:
            self.send_error(400, "bad stream id")
            return

        if stream_id != 0:
            self.send_error(404, "stream not found")
            return

        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        last_seq = -1
        try:
            while True:
                jpeg, seq = self.streamer.wait_for_frame(stream_id, last_seq, timeout=1.0)
                if jpeg is None or seq == last_seq:
                    continue
                last_seq = seq
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(b"Content-Length: %d\r\n\r\n" % len(jpeg))
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as error:
            print("[stream] client error:", error)

    def _serve_recording(self):
        recording_path = self.streamer.latest_recording_path()
        if recording_path is None or not os.path.isfile(recording_path):
            self.send_error(404, "no completed recording")
            return

        filename = os.path.basename(recording_path)
        file_size = os.path.getsize(recording_path)
        self.send_response(200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Disposition", 'attachment; filename="%s"' % filename)
        self.send_header("Content-Length", str(file_size))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            with open(recording_path, "rb") as handle:
                while True:
                    chunk = handle.read(1024 * 1024)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class WebStreamer:
    def __init__(
        self,
        port=STREAM_PORT,
        jpeg_quality=STREAM_JPEG_QUALITY,
        recording_fps=RECORDING_FPS,
        recording_directory=RECORDING_DIRECTORY,
    ):
        self.port = port
        self.jpeg_quality = max(20, min(100, int(jpeg_quality)))
        self.recording_fps = max(1.0, float(recording_fps))
        self.recording_directory = os.path.abspath(recording_directory)
        self.frames = {0: None}
        self.sequence = {0: 0}
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.record_lock = threading.Lock()
        self.recording = False
        self.record_writer = None
        self.record_path = None
        self.record_size = None
        self.record_frames = 0
        self.record_started_at = None
        self.record_error = ""
        self.latest_recording = None
        self.server = None
        self.server_thread = None
        self._start_server()

    def _start_server(self):
        MJPEGHandler.streamer = self
        self.server = ThreadedHTTPServer(("0.0.0.0", self.port), MJPEGHandler)
        self.server_thread = threading.Thread(
            target=self.server.serve_forever,
            daemon=True,
        )
        self.server_thread.start()
        print("[stream] http://localhost:%d" % self.port)

    def start_recording(self):
        with self.record_lock:
            if self.recording:
                return False, "Recording is already active"
            os.makedirs(self.recording_directory, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            milliseconds = int((time.time() * 1000) % 1000)
            self.record_path = os.path.join(
                self.recording_directory,
                "k7c_ball_%s_%03d.mp4" % (timestamp, milliseconds),
            )
            self.record_writer = None
            self.record_size = None
            self.record_frames = 0
            self.record_started_at = None
            self.record_error = ""
            self.recording = True
            print("[record] started %s" % self.record_path)
            return True, "Recording started"

    def _write_recording_frame(self, frame):
        with self.record_lock:
            if not self.recording or frame is None:
                return

            if frame.ndim == 2:
                record_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:
                record_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            else:
                record_frame = frame

            height, width = record_frame.shape[:2]
            if self.record_writer is None:
                self.record_size = (width, height)
                fourcc = cv2.VideoWriter_fourcc(*RECORDING_FOURCC)
                writer = cv2.VideoWriter(
                    self.record_path,
                    fourcc,
                    self.recording_fps,
                    self.record_size,
                    True,
                )
                if not writer.isOpened():
                    writer.release()
                    self.record_error = (
                        "OpenCV cannot open the mp4v encoder; install FFmpeg-enabled OpenCV"
                    )
                    self.recording = False
                    print("[record] error:", self.record_error)
                    return
                self.record_writer = writer
                self.record_started_at = time.perf_counter()

            if (width, height) != self.record_size:
                record_frame = cv2.resize(
                    record_frame,
                    self.record_size,
                    interpolation=cv2.INTER_AREA,
                )

            elapsed = max(0.0, time.perf_counter() - self.record_started_at)
            target_frames = int(elapsed * self.recording_fps) + 1
            copies = max(1, target_frames - self.record_frames)
            copies = min(copies, max(1, int(self.recording_fps)))
            for _ in range(copies):
                self.record_writer.write(record_frame)
            self.record_frames += copies

    def stop_recording(self):
        with self.record_lock:
            if not self.recording and self.record_writer is None:
                return False, self.record_error or "Recording is not active"

            self.recording = False
            if self.record_writer is not None:
                self.record_writer.release()
                self.record_writer = None

            completed_path = self.record_path
            self.record_path = None
            if (
                self.record_frames > 0
                and completed_path is not None
                and os.path.isfile(completed_path)
                and os.path.getsize(completed_path) > 0
            ):
                self.latest_recording = completed_path
                print(
                    "[record] saved %s frames=%d"
                    % (completed_path, self.record_frames)
                )
                return True, "Recording saved"

            self.record_error = self.record_error or "No video frames were recorded"
            if completed_path and os.path.isfile(completed_path):
                try:
                    os.remove(completed_path)
                except OSError:
                    pass
            return False, self.record_error

    def recording_status(self):
        with self.record_lock:
            latest_ready = (
                not self.recording
                and self.latest_recording is not None
                and os.path.isfile(self.latest_recording)
            )
            return {
                "recording": self.recording,
                "frames": self.record_frames,
                "seconds": self.record_frames / self.recording_fps,
                "filename": (
                    os.path.basename(self.latest_recording)
                    if latest_ready else ""
                ),
                "ready": latest_ready,
                "error": self.record_error,
            }

    def latest_recording_path(self):
        with self.record_lock:
            if self.recording or self.latest_recording is None:
                return None
            return self.latest_recording

    def publish(self, stream_id, frame):
        if stream_id != 0:
            raise ValueError("stream_id must be 0")

        if frame is None:
            jpeg = None
        elif isinstance(frame, (bytes, bytearray, memoryview)):
            jpeg = bytes(frame)
        else:
            self._write_recording_frame(frame)
            ok, encoded = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)],
            )
            if not ok:
                return False
            jpeg = encoded.tobytes()

        with self.condition:
            self.frames[stream_id] = jpeg
            self.sequence[stream_id] += 1
            self.condition.notify_all()
        return True

    def wait_for_frame(self, stream_id, last_seq, timeout=1.0):
        with self.condition:
            if self.sequence[stream_id] == last_seq:
                self.condition.wait(timeout)
            return self.frames[stream_id], self.sequence[stream_id]

    def stop(self):
        if self.recording or self.record_writer is not None:
            self.stop_recording()
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None


class BallTracker:
    def __init__(
        self,
        frame_width,
        frame_height,
        roi_base_y=ROI_BASE_Y,
        roi_base_h=ROI_BASE_H,
        search_x_margin=SEARCH_X_MARGIN,
        search_x_min=SEARCH_X_MIN,
        search_x_max=SEARCH_X_MAX,
        search_y_min=SEARCH_Y_MIN,
        search_y_max=None,
        centerline_y=None,
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.center_x = frame_width // 2
        self.center_y = frame_height // 2
        self.roi_x, self.roi_y, self.roi_w, self.roi_h = scale_roi(
            frame_width, frame_height, roi_base_y, roi_base_h
        )
        self.scale_x = float(frame_width) / float(BASE_FRAME_WIDTH)
        self.scale_y = float(frame_height) / float(BASE_FRAME_HEIGHT)
        self.scale = (self.scale_x + self.scale_y) * 0.5
        self.area_scale = max(0.05, self.scale_x * self.scale_y)
        self.ball_diameter_pixels = BALL_DIAMETER_PIXELS_BASE * self.scale_x
        self.mm_per_pixel = BALL_DIAMETER_MM / self.ball_diameter_pixels

        self.blackhat_kernel_size = scaled_odd(BLACKHAT_KERNEL_SIZE, self.scale)
        self.blackhat_blur_kernel = scaled_odd(BLACKHAT_BLUR_KERNEL, self.scale)
        self.background_blur_kernel = scaled_odd(BACKGROUND_BLUR_KERNEL, self.scale)
        self.illumination_kernel_x = scaled_odd(
            ILLUMINATION_KERNEL_X, self.scale_x
        )
        self.illumination_kernel_y = scaled_odd(
            ILLUMINATION_KERNEL_Y, self.scale_y
        )
        self.high_contrast_kernel_x = scaled_odd(
            HIGH_CONTRAST_KERNEL_X, self.scale_x
        )
        self.high_contrast_kernel_y = scaled_odd(
            HIGH_CONTRAST_KERNEL_Y, self.scale_y
        )
        self.search_x_margin = scaled_int(search_x_margin, self.scale_x, 0)
        self.search_x_min = scaled_int(search_x_min, self.scale_x, 0)
        self.search_x_max = scaled_int(search_x_max, self.scale_x, 0)
        self.search_y_min = scaled_int(search_y_min, self.scale_y, 0)
        if search_y_max is None:
            search_y_max = roi_base_h
        self.search_y_max = scaled_int(search_y_max, self.scale_y)
        self.track_search_x = scaled_int(TRACK_SEARCH_X, self.scale_x)
        self.track_search_y = scaled_int(TRACK_SEARCH_Y, self.scale_y)
        self.blackhat_center_y_offset = int(BLACKHAT_CENTER_Y_OFFSET * self.scale_y)
        self.fallback_radius = scaled_int(FALLBACK_RADIUS, self.scale)
        self.min_radius = scaled_int(MIN_RADIUS, self.scale)
        self.max_radius = scaled_int(MAX_RADIUS, self.scale)
        self.hough_min_distance = scaled_int(HOUGH_MIN_DISTANCE, self.scale)
        self.gaussian_kernel = scaled_odd(GAUSSIAN_KERNEL, self.scale)
        self.max_anchor_distance = scaled_int(MAX_ANCHOR_DISTANCE, self.scale)
        self.edge_margin = scaled_int(EDGE_MARGIN, self.scale)
        self.max_center_jump = scaled_int(MAX_CENTER_JUMP, self.scale_x)
        self.fast_motion_threshold = FAST_MOTION_THRESHOLD * self.scale_x
        self.box_margin = scaled_int(BOX_MARGIN, self.scale)

        self.min_response_area = max(2.0, MIN_BACKGROUND_AREA * self.area_scale)
        self.max_response_area = max(
            self.min_response_area + 1.0,
            MAX_BACKGROUND_AREA * self.area_scale,
        )
        self.min_response_size = scaled_int(MIN_BACKGROUND_SIZE, self.scale)
        self.max_response_width = scaled_int(MAX_BACKGROUND_WIDTH, self.scale_x)
        self.max_response_height = scaled_int(MAX_BACKGROUND_HEIGHT, self.scale_y)
        self.min_ball_equivalent_diameter = (
            MIN_BALL_EQUIVALENT_DIAMETER * self.scale
        )
        self.max_ball_equivalent_diameter = (
            MAX_BALL_EQUIVALENT_DIAMETER * self.scale
        )
        if centerline_y is None:
            self.path_y = (self.roi_h - 1) * 0.5
        else:
            self.path_y = float(scaled_int(centerline_y, self.scale_y, 0))
        self.centerline_half_height = scaled_int(
            CENTERLINE_HALF_HEIGHT, self.scale_y, 1
        )
        self.centerline_pixel_gap = scaled_int(
            CENTERLINE_PIXEL_GAP, self.scale_x, 1
        )
        self.centerline_smooth_kernel = scaled_odd(
            CENTERLINE_SMOOTH_KERNEL, self.scale_x
        )
        self.centerline_outside_band = scaled_int(
            CENTERLINE_OUTSIDE_BAND, self.scale_x, 2
        )
        self.path_tolerance = scaled_int(BALL_PATH_TOLERANCE, self.scale_y)
        self.track_path_tolerance = scaled_int(TRACK_PATH_TOLERANCE, self.scale_y)
        self.background_freeze_margin = scaled_int(BACKGROUND_FREEZE_MARGIN, self.scale)

        self.blackhat_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.blackhat_kernel_size, self.blackhat_kernel_size),
        )
        open_size = scaled_odd(BACKGROUND_OPEN_KERNEL, self.scale)
        close_size = scaled_odd(BACKGROUND_CLOSE_KERNEL, self.scale)
        self.background_open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (open_size, open_size),
        )
        self.background_close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (close_size, close_size),
        )

        self.background = None
        self.high_contrast_response = None
        self.background_warmup = BACKGROUND_WARMUP_FRAMES
        self.background_decay_image = None
        self.calibrating = False
        self.last_edge_threshold = 0.0
        self.last_edge_score = 0.0
        self.last_dark_contrast = 0.0
        self.track = {
            "valid": False,
            "cx": 0.0,
            "cy": 0.0,
            "radius": 0.0,
            "strength": 0.0,
            "vx": 0.0,
            "vy": 0.0,
            "last_measure_x": 0.0,
            "last_measure_y": 0.0,
            "hits": 0,
            "lost": 0,
        }
        self.frame_count = 0
        self.fps = 0.0
        self.saturation_percent = 0.0

    def validate(self):
        if self.roi_x < 0 or self.roi_y < 0:
            raise ValueError("invalid ROI")
        if self.roi_w <= 0 or self.roi_h <= 0:
            raise ValueError("invalid ROI size")
        if self.roi_x + self.roi_w > self.frame_width:
            raise ValueError("ROI exceeds frame width")
        if self.roi_y + self.roi_h > self.frame_height:
            raise ValueError("ROI exceeds frame height")
        if self.search_y_min < 0 or self.search_y_max > self.roi_h:
            raise ValueError("search y band exceeds ROI height")
        if self.search_y_min >= self.search_y_max:
            raise ValueError("search-y-min must be smaller than search-y-max")
        if self.search_x_margin < 0 or self.search_x_margin * 2 >= self.roi_w:
            raise ValueError("search-x-margin leaves no searchable ROI width")
        if self.search_x_min < 0 or self.search_x_max > self.roi_w:
            raise ValueError("centerline search x range exceeds ROI width")
        if self.search_x_min >= self.search_x_max:
            raise ValueError("search-x-min must be smaller than search-x-max")
        if not 0 <= self.path_y < self.roi_h:
            raise ValueError("centerline-y exceeds ROI height")

    @staticmethod
    def patch_mean(gray, x0, y0, x1, y1):
        patch = gray[y0:y1, x0:x1]
        return float(cv2.mean(patch)[0]) if patch.size else 0.0

    def local_contrast(self, gray, cx, cy, radius):
        inner_r = max(2, radius)
        outer_r = radius + max(3, radius // 2)
        ix0 = max(0, cx - inner_r)
        iy0 = max(0, cy - inner_r)
        ix1 = min(self.roi_w, cx + inner_r + 1)
        iy1 = min(self.roi_h, cy + inner_r + 1)
        ox0 = max(0, cx - outer_r)
        oy0 = max(0, cy - outer_r)
        ox1 = min(self.roi_w, cx + outer_r + 1)
        oy1 = min(self.roi_h, cy + outer_r + 1)
        inner_mean = self.patch_mean(gray, ix0, iy0, ix1, iy1)
        outer_mean = self.patch_mean(gray, ox0, oy0, ox1, oy1)
        inner_area = (ix1 - ix0) * (iy1 - iy0)
        outer_area = (ox1 - ox0) * (oy1 - oy0)
        ring_area = outer_area - inner_area
        if ring_area <= 0:
            return 0.0
        ring_mean = (outer_mean * outer_area - inner_mean * inner_area) / ring_area
        return ring_mean - inner_mean

    def _contour_candidate(self, gray, response, contour, origin_x=0, origin_y=0):
        area = float(cv2.contourArea(contour))
        if area < self.min_response_area or area > self.max_response_area:
            return None

        bx, by, bw, bh = cv2.boundingRect(contour)
        response_h, response_w = response.shape[:2]
        touches_artificial_edge = (
            (bx <= 0 and origin_x > 0)
            or (by <= 0 and origin_y > 0)
            or (
                bx + bw >= response_w
                and origin_x + response_w < self.roi_w
            )
            or (
                by + bh >= response_h
                and origin_y + response_h < self.roi_h
            )
        )
        if touches_artificial_edge:
            return None
        if (
            bw < self.min_response_size
            or bh < self.min_response_size
            or bw > self.max_response_width
            or bh > self.max_response_height
        ):
            return None
        if bh >= self.roi_h - 2 or bw >= max(8, self.roi_w // 8):
            return None

        aspect = float(bw) / float(max(1, bh))
        frame_left = origin_x + bx
        frame_right = frame_left + bw
        touches_x_edge = frame_left <= 1 or frame_right >= self.roi_w - 1
        equivalent_diameter = math.sqrt(4.0 * area / math.pi)
        minimum_diameter = self.min_ball_equivalent_diameter
        if touches_x_edge:
            minimum_diameter *= 0.65
        if (
            equivalent_diameter < minimum_diameter
            or equivalent_diameter > self.max_ball_equivalent_diameter
        ):
            return None

        min_aspect = 0.20 if touches_x_edge else MIN_BACKGROUND_ASPECT
        max_aspect = 4.00 if touches_x_edge else MAX_BACKGROUND_ASPECT
        if aspect < min_aspect or aspect > max_aspect:
            return None

        fill_ratio = area / float(max(1, bw * bh))
        if fill_ratio < MIN_BACKGROUND_FILL:
            return None

        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0.0:
            return None
        contour_circularity = min(
            1.0, 4.0 * math.pi * area / (perimeter * perimeter)
        )
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        hull_perimeter = float(cv2.arcLength(hull, True))
        if hull_area <= 0.0 or hull_perimeter <= 0.0:
            return None
        hull_circularity = min(
            1.0,
            4.0 * math.pi * hull_area / (hull_perimeter * hull_perimeter),
        )
        solidity = min(1.0, area / hull_area)
        _, enclosing_radius = cv2.minEnclosingCircle(contour)
        circle_area = math.pi * max(1.0, enclosing_radius * enclosing_radius)
        circle_fill = min(1.0, area / circle_area)

        edge_relax = 0.60 if touches_x_edge else 1.0
        if (
            contour_circularity < MIN_CONTOUR_CIRCULARITY * edge_relax
            or hull_circularity < MIN_HULL_CIRCULARITY * edge_relax
            or circle_fill < MIN_CIRCLE_FILL * edge_relax
            or solidity < MIN_SOLIDITY * edge_relax
        ):
            return None

        patch = response[by:by + bh, bx:bx + bw]
        mean_response = float(cv2.mean(patch)[0])
        _, strength, _, peak_location = cv2.minMaxLoc(patch)
        gx0 = origin_x + bx
        gy0 = origin_y + by
        gray_patch = gray[gy0:gy0 + bh, gx0:gx0 + bw]
        _, gray_stddev = cv2.meanStdDev(gray_patch)
        texture_std = float(gray_stddev[0][0])
        gray_min, gray_max, _, _ = cv2.minMaxLoc(gray_patch)
        dynamic_range = float(gray_max - gray_min)
        if texture_std < MIN_TEXTURE_STD:
            return None

        high_contrast_mean = 0.0
        high_contrast_peak = 0.0
        if self.high_contrast_response is not None:
            contrast_patch = self.high_contrast_response[
                gy0:gy0 + bh,
                gx0:gx0 + bw,
            ]
            if contrast_patch.size:
                high_contrast_mean = float(cv2.mean(contrast_patch)[0])
                _, high_contrast_peak, _, _ = cv2.minMaxLoc(contrast_patch)
                contrast_relax = 0.75 if self.track["valid"] else 1.0
                if (
                    high_contrast_peak < MIN_HIGH_CONTRAST_PEAK * contrast_relax
                    or high_contrast_mean < MIN_HIGH_CONTRAST_MEAN * contrast_relax
                ):
                    return None

        box_cx = origin_x + bx + bw // 2
        cy = origin_y + by + bh // 2
        peak_x = origin_x + bx + int(peak_location[0])
        cx = int((box_cx * 3 + peak_x) / 4)
        radius = max(2, min(self.max_radius, int((bw + bh) / 4)))
        contrast = self.local_contrast(gray, cx, cy, radius)
        if contrast < MIN_CANDIDATE_CONTRAST:
            return None

        path_tolerance = (
            self.track_path_tolerance if self.track["valid"] else self.path_tolerance
        )
        path_error = abs(cy - self.path_y)
        if path_error > path_tolerance:
            return None

        predicted_x = self.track["cx"] + self.track["vx"]
        predicted_y = self.track["cy"] + self.track["vy"]
        tracking_bonus = 0.0
        if self.track["valid"]:
            distance = abs(cx - predicted_x) + abs(cy - predicted_y)
            motion_gate = self.max_center_jump + min(
                self.track_search_x,
                int(abs(self.track["vx"]) * 1.5),
            )
            if distance > motion_gate:
                return None
            tracking_bonus = (motion_gate - distance) * 2.5

        roundness = 1.0 - abs(bw - bh) / float(max(bw, bh))
        expected_area = math.pi * (self.ball_diameter_pixels * 0.5) ** 2
        size_match = min(area, expected_area) / max(area, expected_area)
        shape_score = (
            contour_circularity * 45.0
            + hull_circularity * 45.0
            + circle_fill * 45.0
            + solidity * 25.0
            + roundness * 25.0
            + size_match * 50.0
        )
        score = (
            float(strength) * 2.5
            + mean_response * 2.0
            + contrast * 6.0
            + min(texture_std, 40.0) * 2.0
            + min(dynamic_range, 100.0) * 0.4
            + min(high_contrast_peak, 80.0) * 2.0
            + min(high_contrast_mean, 40.0) * 4.0
            + shape_score
            + tracking_bonus
            - path_error * 2.5
        )
        return (score, cx, cy, radius, float(strength))

    def detect_centerline_ball(self, gray):
        x0 = max(0, self.search_x_min)
        x1 = min(self.roi_w, self.search_x_max)
        center_y = int(self.path_y + 0.5)
        y0 = max(0, center_y - self.centerline_half_height)
        y1 = min(self.roi_h, center_y + self.centerline_half_height + 1)
        if x1 - x0 < 16 or y1 <= y0:
            return None

        stripe = gray[y0:y1, x0:x1]
        raw_profile = np.median(stripe, axis=0).astype(np.float32)
        profile = cv2.GaussianBlur(
            raw_profile.reshape(1, -1),
            (self.centerline_smooth_kernel, 1),
            0,
        ).reshape(-1)
        profile = profile.astype(np.float32, copy=False)

        gap = self.centerline_pixel_gap
        length = int(profile.size)
        if length <= gap * 2 + 4:
            return None
        gradient = np.zeros(length, dtype=np.float32)
        gradient[gap:length - gap] = (
            profile[:length - gap * 2] - profile[gap * 2:]
        )
        valid_gradient = np.abs(gradient[gap:length - gap])
        noise = float(np.median(valid_gradient)) if valid_gradient.size else 0.0
        edge_threshold = max(
            CENTERLINE_EDGE_FLOOR,
            noise * CENTERLINE_NOISE_MULTIPLIER,
        )
        self.last_edge_threshold = edge_threshold
        self.last_edge_score = 0.0
        self.last_dark_contrast = 0.0

        min_diameter = max(3, int(self.min_ball_equivalent_diameter + 0.5))
        max_diameter = max(
            min_diameter,
            int(self.max_ball_equivalent_diameter + 0.5),
        )
        outside_band = self.centerline_outside_band
        expected_diameter = max(1.0, self.ball_diameter_pixels)
        best = None
        best_score = -1000000.0

        left_start = gap + outside_band
        left_stop = length - gap - outside_band - min_diameter
        for left in range(left_start, max(left_start, left_stop)):
            left_edge = float(gradient[left])
            if left_edge < edge_threshold:
                continue

            right_start = left + min_diameter
            right_stop = min(
                length - gap - outside_band,
                left + max_diameter + 1,
            )
            for right in range(right_start, right_stop):
                right_edge = float(-gradient[right])
                if right_edge < edge_threshold:
                    continue

                edge_balance = min(left_edge, right_edge) / max(
                    left_edge, right_edge
                )
                if edge_balance < CENTERLINE_MIN_EDGE_BALANCE:
                    continue

                inside = profile[left + gap:right - gap + 1]
                outside_left = profile[
                    left - gap - outside_band:left - gap
                ]
                outside_right = profile[
                    right + gap:right + gap + outside_band
                ]
                if (
                    inside.size == 0
                    or outside_left.size == 0
                    or outside_right.size == 0
                ):
                    continue

                inside_dark = float(np.percentile(inside, 35.0))
                outside_level = float(
                    np.median(np.concatenate((outside_left, outside_right)))
                )
                dark_contrast = outside_level - inside_dark
                if dark_contrast < CENTERLINE_MIN_DARK_CONTRAST:
                    continue
                raw_inside = raw_profile[left:right + 1]
                dark_run_pixels = int(
                    np.count_nonzero(
                        raw_inside
                        <= outside_level - CENTERLINE_MIN_DARK_CONTRAST
                    )
                )
                minimum_dark_run = int(
                    self.min_ball_equivalent_diameter
                    * CENTERLINE_MIN_DARK_RUN_RATIO
                    + 0.5
                )
                if dark_run_pixels < minimum_dark_run:
                    continue

                diameter = right - left
                cx = x0 + (left + right) * 0.5
                tracking_bonus = 0.0
                if self.track["valid"]:
                    predicted_x = self.track["cx"] + self.track["vx"]
                    distance = abs(cx - predicted_x)
                    motion_gate = self.max_center_jump + min(
                        self.track_search_x,
                        int(abs(self.track["vx"]) * 1.5),
                    )
                    if distance > motion_gate:
                        continue
                    tracking_bonus = max(0.0, motion_gate - distance) * 0.35

                size_match = max(
                    0.0,
                    1.0 - abs(diameter - expected_diameter) / expected_diameter,
                )
                score = (
                    left_edge
                    + right_edge
                    + dark_contrast * 2.0
                    + edge_balance * 12.0
                    + size_match * 10.0
                    + tracking_bonus
                )
                if score > best_score:
                    best_score = score
                    strength = min(
                        100.0,
                        (left_edge + right_edge) * 0.5 + dark_contrast,
                    )
                    best = (
                        int(cx + 0.5),
                        center_y,
                        max(2, int(diameter * 0.5 + 0.5)),
                        strength,
                    )
                    self.last_edge_score = left_edge + right_edge
                    self.last_dark_contrast = dark_contrast

        return best

    def background_peak(self, gray):
        response = cv2.subtract(self.background, gray)
        response = cv2.GaussianBlur(
            response,
            (self.background_blur_kernel, self.background_blur_kernel),
            0,
        )
        _, maximum, _, _ = cv2.minMaxLoc(response)
        minimum = (
            MIN_TRACKED_BACKGROUND_RESPONSE
            if self.track["valid"] else MIN_BACKGROUND_RESPONSE
        )
        if maximum < minimum:
            return None

        threshold_value = max(
            int(minimum),
            int(float(maximum) * BACKGROUND_THRESHOLD_RATIO),
        )
        _, response_mask = cv2.threshold(
            response,
            threshold_value,
            255,
            cv2.THRESH_BINARY,
        )
        response_mask = cv2.morphologyEx(
            response_mask,
            cv2.MORPH_OPEN,
            self.background_open_kernel,
        )
        response_mask = cv2.morphologyEx(
            response_mask,
            cv2.MORPH_CLOSE,
            self.background_close_kernel,
        )
        contours, _ = cv2.findContours(
            response_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        best = None
        best_score = -1000000.0
        for contour in contours:
            item = self._contour_candidate(gray, response, contour)
            if item is not None and item[0] > best_score:
                best_score = item[0]
                best = (item[1], item[2], item[3], item[4])
        return best

    def update_bright_background(self, gray):
        frozen = None
        fx0 = fy0 = fx1 = fy1 = 0
        if self.track["valid"]:
            freeze_radius = int(
                self.track["radius"] + self.background_freeze_margin
            )
            fx0 = max(0, int(self.track["cx"]) - freeze_radius)
            fy0 = max(0, int(self.track["cy"]) - freeze_radius)
            fx1 = min(self.roi_w, int(self.track["cx"]) + freeze_radius + 1)
            fy1 = min(self.roi_h, int(self.track["cy"]) + freeze_radius + 1)
            frozen = self.background[fy0:fy1, fx0:fx1].copy()

        decayed = cv2.subtract(self.background, self.background_decay_image)
        updated = cv2.max(decayed, gray)
        if frozen is not None:
            updated[fy0:fy1, fx0:fx1] = frozen
        self.background = updated

    def blackhat_peak(self, gray, x0, y0, x1, y1, minimum):
        if x1 <= x0 or y1 <= y0:
            return None
        patch = gray[y0:y1, x0:x1]
        blackhat = cv2.morphologyEx(
            patch,
            cv2.MORPH_BLACKHAT,
            self.blackhat_kernel,
        )
        response = cv2.GaussianBlur(
            blackhat,
            (self.blackhat_blur_kernel, self.blackhat_blur_kernel),
            0,
        )
        _, response_mask = cv2.threshold(
            response,
            int(minimum),
            255,
            cv2.THRESH_BINARY,
        )
        contours, _ = cv2.findContours(
            response_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        best = None
        best_score = -1000000.0
        for contour in contours:
            item = self._contour_candidate(
                gray,
                response,
                contour,
                origin_x=x0,
                origin_y=y0,
            )
            if item is not None and item[0] > best_score:
                best_score = item[0]
                best = (item[1], item[2], self.fallback_radius, item[4])
        return best

    def detect_dark_spot(self, gray):
        if self.track["valid"]:
            predicted_x = int(self.track["cx"] + self.track["vx"] + 0.5)
            predicted_y = int(self.track["cy"] + self.track["vy"] + 0.5)
            x0 = max(self.search_x_margin, predicted_x - self.track_search_x)
            x1 = min(
                self.roi_w - self.search_x_margin,
                predicted_x + self.track_search_x + 1,
            )
            y0 = max(self.search_y_min, predicted_y - self.track_search_y)
            y1 = min(self.search_y_max, predicted_y + self.track_search_y + 1)
            peak = self.blackhat_peak(
                gray,
                x0,
                y0,
                x1,
                y1,
                MIN_TRACKED_RESPONSE,
            )
            if peak is not None:
                return peak

        return self.blackhat_peak(
            gray,
            self.search_x_margin,
            self.search_y_min,
            self.roi_w - self.search_x_margin,
            self.search_y_max,
            MIN_BLACKHAT_RESPONSE,
        )

    def select_circle(self, gray, circles, anchor):
        if anchor is None or circles is None:
            return anchor
        iterable = circles[0] if len(circles.shape) == 3 else circles
        count = min(len(iterable), MAX_CIRCLES)
        best = None
        best_score = -1000000.0
        for index in range(count):
            cx = int(iterable[index, 0])
            cy = int(iterable[index, 1])
            radius = int(iterable[index, 2])
            if radius < self.min_radius or radius > self.max_radius:
                continue
            if (
                cx - radius <= self.edge_margin
                or cy - radius <= self.edge_margin
                or cx + radius >= self.roi_w - self.edge_margin
                or cy + radius >= self.roi_h - self.edge_margin
            ):
                continue
            anchor_distance = abs(cx - anchor[0]) + abs(cy - anchor[1])
            if anchor_distance > self.max_anchor_distance:
                continue
            contrast = self.local_contrast(gray, cx, cy, radius)
            if contrast < MIN_LOCAL_CONTRAST:
                continue
            score = contrast * 5.0 - anchor_distance
            if self.track["valid"]:
                distance = abs(cx - self.track["cx"]) + abs(cy - self.track["cy"])
                score += self.max_center_jump - min(distance, self.max_center_jump * 2)
            if score > best_score:
                best_score = score
                best = (cx, cy, radius, anchor[3])
        return best if best is not None else anchor

    def mark_missed(self):
        if not self.track["valid"]:
            return
        self.track["lost"] += 1
        if self.track["lost"] <= LOST_HOLD_FRAMES:
            self.track["cx"] += self.track["vx"]
            self.track["cy"] += self.track["vy"]
            self.track["vx"] *= VELOCITY_DECAY
            self.track["vy"] *= VELOCITY_DECAY
            self.track["cx"] = max(0.0, min(self.roi_w - 1.0, self.track["cx"]))
            self.track["cy"] = max(0.0, min(self.roi_h - 1.0, self.track["cy"]))
        else:
            self.track["valid"] = False
            self.track["hits"] = 0

    def update_track(self, candidate):
        if candidate is None:
            self.mark_missed()
            return

        cx, cy, radius, strength = candidate
        if self.track["valid"]:
            predicted_x = self.track["cx"] + self.track["vx"]
            predicted_y = self.track["cy"] + self.track["vy"]
            distance = abs(cx - predicted_x) + abs(cy - predicted_y)
            motion_gate = self.max_center_jump + min(
                self.track_search_x,
                int(abs(self.track["vx"]) * 1.5),
            )
            if distance > motion_gate:
                self.mark_missed()
                return

            elapsed_frames = max(1, self.track["lost"] + 1)
            measured_vx = (cx - self.track["last_measure_x"]) / elapsed_frames
            measured_vy = (cy - self.track["last_measure_y"]) / elapsed_frames
            self.track["vx"] = (
                self.track["vx"] * (1.0 - VELOCITY_ALPHA)
                + measured_vx * VELOCITY_ALPHA
            )
            self.track["vy"] = (
                self.track["vy"] * (1.0 - VELOCITY_ALPHA)
                + measured_vy * VELOCITY_ALPHA
            )
            speed = abs(measured_vx) + abs(measured_vy)
            position_alpha = (
                FAST_POSITION_ALPHA
                if speed >= self.fast_motion_threshold
                else POSITION_ALPHA
            )
            self.track["cx"] += (cx - self.track["cx"]) * position_alpha
            self.track["cy"] += (cy - self.track["cy"]) * position_alpha
            self.track["radius"] += (
                radius - self.track["radius"]
            ) * RADIUS_ALPHA
            self.track["last_measure_x"] = float(cx)
            self.track["last_measure_y"] = float(cy)
            self.track["strength"] = strength
            self.track["hits"] += 1
            self.track["lost"] = 0
            if self.track["hits"] >= CONFIRM_FRAMES:
                self.path_y += (cy - self.path_y) * 0.03
            return

        self.track["valid"] = True
        self.track["cx"] = float(cx)
        self.track["cy"] = float(cy)
        self.track["radius"] = float(radius)
        self.track["strength"] = strength
        self.track["vx"] = 0.0
        self.track["vy"] = 0.0
        self.track["last_measure_x"] = float(cx)
        self.track["last_measure_y"] = float(cy)
        self.track["hits"] = 1
        self.track["lost"] = 0

    def current_output(self):
        if not self.track["valid"] or self.track["hits"] < CONFIRM_FRAMES:
            return None
        cx = int(self.track["cx"] + self.track["vx"] * DISPLAY_PREDICTION + 0.5)
        cy = int(self.track["cy"] + self.track["vy"] * DISPLAY_PREDICTION + 0.5)
        cx = max(0, min(self.roi_w - 1, cx))
        cy = max(0, min(self.roi_h - 1, cy))
        radius = max(1, int(self.track["radius"] + 0.5))
        frame_x = self.roi_x + cx
        frame_y = self.roi_y + cy
        dx = frame_x - self.center_x
        dy = frame_y - self.center_y
        dx_mm_float = dx * self.mm_per_pixel
        dx_mm = (
            int(dx_mm_float + 0.5)
            if dx_mm_float >= 0
            else int(dx_mm_float - 0.5)
        )
        confidence = min(1.0, self.track["hits"] / float(CONFIRM_FRAMES + 2))
        confidence *= min(1.0, self.track["strength"] / 24.0)
        return {
            "cx": cx,
            "cy": cy,
            "radius": radius,
            "frame_x": frame_x,
            "frame_y": frame_y,
            "dx": dx,
            "dy": dy,
            "dx_mm": dx_mm,
            "strength": self.track["strength"],
            "confidence": confidence,
        }

    def process(self, frame_bgr, frame_index):
        roi = frame_bgr[
            self.roi_y:self.roi_y + self.roi_h,
            self.roi_x:self.roi_x + self.roi_w,
        ]
        if roi.ndim == 2:
            roi_gray = roi
        else:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        illumination = cv2.GaussianBlur(
            roi_gray,
            (self.illumination_kernel_x, self.illumination_kernel_y),
            0,
        )
        normalized_gray = cv2.addWeighted(
            roi_gray, 1.0, illumination, -1.0, 128.0
        )
        detector_gray = cv2.GaussianBlur(
            normalized_gray,
            (self.background_blur_kernel, self.background_blur_kernel),
            0,
        )
        self.high_contrast_response = None
        preview_gray = cv2.addWeighted(
            normalized_gray,
            PREVIEW_CONTRAST,
            normalized_gray,
            0.0,
            PREVIEW_OFFSET,
        )
        if frame_index % SATURATION_SAMPLE_EVERY == 0:
            _, saturation_mask = cv2.threshold(
                roi_gray,
                SATURATION_THRESHOLD - 1,
                255,
                cv2.THRESH_BINARY,
            )
            self.saturation_percent = (
                100.0 * cv2.countNonZero(saturation_mask) / float(roi_gray.size)
            )

        self.calibrating = False
        candidate = self.detect_centerline_ball(detector_gray)
        self.update_track(candidate)

        output = self.current_output()
        self.frame_count = frame_index
        return preview_gray, output

    def render_roi(self, roi_gray, output, fps, stream_width=ROI_STREAM_WIDTH):
        stream_width = max(160, int(stream_width))
        if stream_width % 2 != 0:
            stream_width += 1
        source_x0 = self.search_x_min
        source_x1 = self.search_x_max
        source_width = max(1, source_x1 - source_x0)
        cropped_roi = roi_gray[:, source_x0:source_x1]
        roi_stream_height = max(
            2,
            int(stream_width * float(self.roi_h) / float(source_width) + 0.5),
        )
        if roi_stream_height % 2 != 0:
            roi_stream_height += 1
        roi_view = cv2.resize(
            cropped_roi,
            (stream_width, roi_stream_height),
            interpolation=(
                cv2.INTER_CUBIC
                if stream_width > source_width else cv2.INTER_AREA
            ),
        )
        scale_x = float(stream_width) / float(source_width)
        scale_y = float(roi_stream_height) / float(self.roi_h)

        target_x = int(
            (self.center_x - self.roi_x - source_x0) * scale_x + 0.5
        )
        if 0 <= target_x < stream_width:
            cv2.line(
                roi_view,
                (target_x, 0),
                (target_x, roi_stream_height - 1),
                150,
                1,
            )
        centerline_y = int(self.path_y * scale_y + 0.5)
        cv2.line(
            roi_view,
            (0, centerline_y),
            (stream_width - 1, centerline_y),
            180,
            1,
        )

        if output is not None:
            draw_cx = int((output["cx"] - source_x0) * scale_x + 0.5)
            draw_cy = int(output["cy"] * scale_y + 0.5)
            draw_radius_x = max(2, int(output["radius"] * scale_x + 0.5))
            draw_radius_y = max(2, int(output["radius"] * scale_y + 0.5))
            x0 = max(0, draw_cx - draw_radius_x - self.box_margin)
            y0 = max(0, draw_cy - draw_radius_y - self.box_margin)
            x1 = min(stream_width - 1, draw_cx + draw_radius_x + self.box_margin)
            y1 = min(
                roi_stream_height - 1,
                draw_cy + draw_radius_y + self.box_margin,
            )
            cv2.rectangle(roi_view, (x0, y0), (x1, y1), 255, 2)
            cv2.drawMarker(
                roi_view,
                (draw_cx, draw_cy),
                0,
                cv2.MARKER_CROSS,
                max(4, min(draw_radius_x, draw_radius_y)),
                1,
            )

        text = "FPS:%.1f x=%s D:%d E:%d C:%d Q:%d SAT:%d%%" % (
            fps,
            "%dmm" % output["dx_mm"] if output is not None else "N",
            int(output["radius"] * 2 + 0.5) if output is not None else 0,
            int(self.last_edge_score + 0.5),
            int(self.last_dark_contrast + 0.5),
            int(output["confidence"] * 100) if output is not None else 0,
            int(self.saturation_percent + 0.5),
        )
        cv2.putText(
            roi_view,
            text,
            (6, min(20, roi_stream_height - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            255,
            1,
            cv2.LINE_AA,
        )
        return roi_view

    def print_status(self, output):
        if self.calibrating:
            print(
                "calibrating=%d/%d sat=%.1f%% fps=%.1f"
                % (
                    self.background_warmup,
                    BACKGROUND_WARMUP_FRAMES,
                    self.saturation_percent,
                    self.fps,
                )
            )
        elif output is None:
            print(
                "ball=(-1,-1) x=N/A v=(%.1f,%.1f) lost=%d edge=%.1f dark=%.1f thr=%.1f sat=%.1f%% fps=%.1f"
                % (
                    self.track["vx"],
                    self.track["vy"],
                    self.track["lost"],
                    self.last_edge_score,
                    self.last_dark_contrast,
                    self.last_edge_threshold,
                    self.saturation_percent,
                    self.fps,
                )
            )
        else:
            print(
                "ball=(%d,%d) x=%dmm err_px=(%d,%d) v=(%.1f,%.1f) resp=%.1f d=%.1f edge=%.1f dark=%.1f thr=%.1f q=%d sat=%.1f%% fps=%.1f"
                % (
                    output["frame_x"],
                    output["frame_y"],
                    output["dx_mm"],
                    output["dx"],
                    output["dy"],
                    self.track["vx"],
                    self.track["vy"],
                    self.track["strength"],
                    output["radius"] * 2.0,
                    self.last_edge_score,
                    self.last_dark_contrast,
                    self.last_edge_threshold,
                    int(output["confidence"] * 100),
                    self.saturation_percent,
                    self.fps,
                )
            )


def open_uart(enabled, port, baud):
    if not enabled:
        return None
    if serial is None:
        raise RuntimeError("pyserial is not installed")
    ser = serial.Serial(
        port=port,
        baudrate=baud,
        timeout=0,
        write_timeout=0,
    )
    print("[uart] opened %s @ %d" % (port, baud))
    return ser


def send_uart(ser, output):
    if ser is None:
        return
    line = "N\n" if output is None else "%d\n" % output["dx_mm"]
    try:
        ser.write(line.encode("utf-8"))
    except Exception as error:
        print("[uart] write error:", error)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX,
                        help="V4L2 camera index, use -1 for auto detect")
    parser.add_argument("--stream", action="store_true", default=False,
                        help="enable LAN MJPEG preview")
    parser.add_argument("--port", type=int, default=STREAM_PORT)
    parser.add_argument("--width", type=int, default=REQUESTED_WIDTH)
    parser.add_argument("--height", type=int, default=REQUESTED_HEIGHT)
    parser.add_argument("--fps", type=int, default=REQUESTED_FPS)
    parser.add_argument("--exposure", type=float, default=None,
                        help="manual camera exposure; backend-specific value")
    parser.add_argument("--gain", type=float, default=None,
                        help="manual camera gain; backend-specific value")
    parser.add_argument("--brightness", type=float, default=None,
                        help="manual camera brightness; backend-specific value")
    parser.add_argument("--stream-fps", type=float, default=STREAM_TARGET_FPS,
                        help="LAN preview and MP4 recording frame rate")
    parser.add_argument("--stream-width", type=int, default=ROI_STREAM_WIDTH,
                        help="rendered grayscale ROI width; larger is clearer but uses more CPU/network")
    parser.add_argument("--jpeg-quality", type=int, default=STREAM_JPEG_QUALITY,
                        help="MJPEG quality from 20 to 100")
    parser.add_argument("--record-dir", default=RECORDING_DIRECTORY,
                        help="directory used for completed MP4 recordings")
    parser.add_argument("--roi-y", type=int, default=ROI_BASE_Y,
                        help="ROI top y in the base 640x480 frame")
    parser.add_argument("--roi-h", type=int, default=ROI_BASE_H,
                        help="ROI height in the base 640x480 frame")
    parser.add_argument("--search-x-margin", type=int, default=SEARCH_X_MARGIN,
                        help="legacy symmetric x margin")
    parser.add_argument("--search-x-min", type=int, default=SEARCH_X_MIN,
                        help="centerline scan left edge inside the base ROI")
    parser.add_argument("--search-x-max", type=int, default=SEARCH_X_MAX,
                        help="centerline scan right edge inside the base ROI")
    parser.add_argument("--centerline-y", type=int, default=None,
                        help="horizontal scan line inside the base ROI; defaults to ROI center")
    parser.add_argument("--search-y-min", type=int, default=SEARCH_Y_MIN,
                        help="search band top y in the base 640x480 ROI")
    parser.add_argument("--search-y-max", type=int, default=None,
                        help="search band bottom y in the base 640x480 ROI; defaults to ROI height")
    parser.add_argument("--log-every", type=int, default=0,
                        help="print one status line every N frames; 0 disables logging")
    parser.add_argument("--uart", action="store_true", default=UART_ENABLED)
    parser.add_argument("--uart-port", default=UART_PORT)
    parser.add_argument("--uart-baud", type=int, default=UART_BAUD)
    return parser.parse_args()


def main():
    args = parse_args()
    cap = None
    grabber = None
    streamer = None
    uart = None

    try:
        requested_camera = None if args.camera < 0 else args.camera
        cap, camera_index, frame_width, frame_height = open_camera(
            requested_camera,
            args.width,
            args.height,
            args.fps,
            exposure=args.exposure,
            gain=args.gain,
            brightness=args.brightness,
        )

        tracker = BallTracker(
            frame_width,
            frame_height,
            roi_base_y=args.roi_y,
            roi_base_h=args.roi_h,
            search_x_margin=args.search_x_margin,
            search_x_min=args.search_x_min,
            search_x_max=args.search_x_max,
            search_y_min=args.search_y_min,
            search_y_max=args.search_y_max,
            centerline_y=args.centerline_y,
        )
        tracker.validate()

        if args.stream:
            recording_fps = args.stream_fps if args.stream_fps > 0 else args.fps
            streamer = WebStreamer(
                port=args.port,
                jpeg_quality=args.jpeg_quality,
                recording_fps=recording_fps,
                recording_directory=args.record_dir,
            )
        uart = open_uart(args.uart, args.uart_port, args.uart_baud)

        if streamer is not None:
            print("[main] open http://<board-ip>:%d" % args.port)
        print(
            "[main] ROI x=%d y=%d w=%d h=%d"
            % (tracker.roi_x, tracker.roi_y, tracker.roi_w, tracker.roi_h)
        )
        print(
            "[main] camera=%d size=%dx%d ball_mm_per_px=%.4f"
            % (camera_index, frame_width, frame_height, tracker.mm_per_pixel)
        )
        print(
            "[main] centerline y=%.1f scan_x=%d..%d diameter=%d..%d"
            % (
                tracker.path_y,
                tracker.search_x_min,
                tracker.search_x_max,
                int(tracker.min_ball_equivalent_diameter + 0.5),
                int(tracker.max_ball_equivalent_diameter + 0.5),
            )
        )

        grabber = FrameGrabber(cap)
        grabber.start()

        last_time = time.perf_counter()
        last_stream_time = 0.0
        stream_interval = 0.0 if args.stream_fps <= 0 else 1.0 / args.stream_fps
        last_capture_seq = 0
        frame_index = 0

        while True:
            frame, capture_seq = grabber.wait_latest(last_capture_seq, timeout=0.2)
            if frame is None or capture_seq == last_capture_seq:
                continue
            last_capture_seq = capture_seq

            now = time.perf_counter()
            dt = now - last_time
            if dt > 0:
                tracker.fps = tracker.fps * 0.8 + (1.0 / dt) * 0.2 if tracker.fps > 0 else (1.0 / dt)
            last_time = now

            roi_gray, output = tracker.process(frame, frame_index)
            if streamer is not None and now - last_stream_time >= stream_interval:
                preview_roi = tracker.render_roi(
                    roi_gray,
                    output,
                    tracker.fps,
                    stream_width=args.stream_width,
                )
                streamer.publish(0, preview_roi)
                last_stream_time = now

            send_uart(uart, output)
            if args.log_every > 0 and frame_index % args.log_every == 0:
                tracker.print_status(output)

            frame_index += 1

            if args.log_every > 0 and frame_index % (args.log_every * 4) == 0:
                gc.collect()

    except KeyboardInterrupt:
        print("[main] stopped by user")
    except BaseException as error:
        print("[main] error:", error)
        raise
    finally:
        if grabber is not None:
            grabber.stop()
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        if streamer is not None:
            streamer.stop()
        if uart is not None:
            try:
                uart.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
