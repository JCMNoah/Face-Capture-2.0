from pathlib import Path
import cv2
import numpy as np
import mss
import time
from collections import deque
import threading
import pyvirtualcam
from pynput import keyboard

from detection.yolo_detector import YOLODetector
from segmentation.face_segmentor import FaceSegmentor

def overlay_rgba(base_img, overlay_img, position):
    x, y = position
    h, w = overlay_img.shape[:2]

    # Check bounds
    if y + h > base_img.shape[0] or x + w > base_img.shape[1]:
        return base_img

    # Split channels
    overlay_rgb = overlay_img[:, :, :3].astype(np.float32)
    alpha = overlay_img[:, :, 3].astype(np.float32) / 255.0
    alpha = np.expand_dims(alpha, axis=2)

    # Region to blend
    base_region = base_img[y:y+h, x:x+w].astype(np.float32)

    # Blend and convert back to uint8
    blended = cv2.addWeighted(overlay_rgb, 1.0, base_region, 0.0, 0)  # start with overlay
    blended = overlay_rgb * alpha + base_region * (1 - alpha)
    base_img[y:y+h, x:x+w] = blended.astype(np.uint8)

    return base_img


def average_bbox(buffer):
    if not buffer:
        return None
    avg_x1 = int(np.mean([b["top_left"][0] for b in buffer]))
    avg_y1 = int(np.mean([b["top_left"][1] for b in buffer]))
    avg_x2 = int(np.mean([b["bottom_right"][0] for b in buffer]))
    avg_y2 = int(np.mean([b["bottom_right"][1] for b in buffer]))
    center = ((avg_x1 + avg_x2) // 2, (avg_y1 + avg_y2) // 2)
    return {
        "top_left": (avg_x1, avg_y1),
        "bottom_right": (avg_x2, avg_y2),
        "center": center,
        "found": True
    }

class WebcamCapture(threading.Thread):
    def __init__(self, segmentor):
        super().__init__()
        self.segmentor = segmentor
        self.running = True
        self.frame = None
        self.masked = None
        self.lock = threading.Lock()
        # PERFORMANCE OPTIMIZATION: Reduce segmentation frequency for better performance
        self.segmentation_interval = 4  # Increased back to every 4th frame for smoother camera
        self.counter = 0
        self.daemon = True  # Make thread daemon for cleaner shutdown

    def run(self):
        # PERFORMANCE OPTIMIZATION: Use DirectShow backend for better Windows performance
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        # BALANCED SETTINGS: Good quality + performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency

        # Simple camera settings - try minimal approach for color
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # Use MJPEG

        # Try basic auto settings
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Enable auto exposure
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus

        print("✅ Camera initialized with default color settings")

        # Warm up camera and flush initial frames
        print("🎥 Warming up camera...")
        for _ in range(10):
            cap.read()
        print("✅ Camera ready")

        while self.running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.001)  # Small delay to prevent CPU spinning
                continue

            # QUALITY IMPROVEMENT: Process every frame instead of skipping
            # We can afford this now that YOLO is threaded
            self.counter += 1
            if self.counter % self.segmentation_interval == 0:
                full_frame, masked = self.segmentor.process_frame(frame)
                with self.lock:
                    self.frame = full_frame
                    self.masked = masked
            else:
                with self.lock:
                    self.frame = frame

        cap.release()

    def get_frames(self):
        with self.lock:
            return self.frame, self.masked

    def stop(self):
        self.running = False


class YOLODetectionThread(threading.Thread):
    def __init__(self, detector):
        super().__init__()
        self.detector = detector
        self.running = True
        self.current_frame = None
        self.latest_detection = {"found": False}
        self.frame_lock = threading.Lock()
        self.detection_lock = threading.Lock()
        self.daemon = True

    def update_frame(self, frame):
        """Update the frame to be processed"""
        with self.frame_lock:
            self.current_frame = frame.copy() if frame is not None else None

    def get_latest_detection(self):
        """Get the most recent detection result"""
        with self.detection_lock:
            return self.latest_detection.copy()

    def run(self):
        while self.running:
            frame_to_process = None
            with self.frame_lock:
                if self.current_frame is not None:
                    frame_to_process = self.current_frame.copy()
                    self.current_frame = None  # Clear to avoid reprocessing

            if frame_to_process is not None:
                detection = self.detector.detect(frame_to_process)
                with self.detection_lock:
                    self.latest_detection = detection
            else:
                time.sleep(0.005)  # 5ms sleep when no frame to process

    def stop(self):
        self.running = False

def main():
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "runs/detect/train7/weights/best.pt"
    detector = YOLODetector(str(model_path), conf_threshold=0.5)
    segmentor = FaceSegmentor()

    debug_mode = False
    quit_flag = False
    print("Press 'd' to toggle debug mode, 'q' to quit.")

    def on_press(key):
        nonlocal debug_mode, quit_flag
        try:
            if key.char == 'q':
                quit_flag = True
            elif key.char == 'd':
                debug_mode = not debug_mode
                print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
        except AttributeError:
            pass

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # PERFORMANCE OPTIMIZATION: Start threaded components
    webcam_thread = WebcamCapture(segmentor)
    webcam_thread.start()

    yolo_thread = YOLODetectionThread(detector)
    yolo_thread.start()

    detection_buffer = deque(maxlen=5)

    crop_x, crop_y, crop_w, crop_h = 700, 200, 512, 512

    # EXACT MATCH: Output size exactly matches crop region
    output_width, output_height = crop_w, crop_h  # 512x512 to match crop
    output_shape = (output_height, output_width, 3)

    # Create green background for chroma key
    green_background = np.full(output_shape, (0, 255, 0), dtype=np.uint8)

    print(f"📺 Virtual camera output: {output_width}x{output_height} (EXACT crop match)")
    print(f"   Crop region: {crop_x},{crop_y} to {crop_x+crop_w},{crop_y+crop_h}")
    print(f"   Output: 0,0 to {output_width},{output_height}")

    # YOLO detection interval for performance
    detection_interval = 2  # Every 2nd frame
    detection_counter = 0
    last_detection = {"found": False}

    with pyvirtualcam.Camera(width=output_width, height=output_height, fps=60) as cam:
        print(f"OBS virtual camera started at {output_width}x{output_height} @ 60fps")
        prev_time = time.time()
        frame_count = 0
        fps = 0

        with mss.mss() as sct:
            # PERFORMANCE OPTIMIZATION: Use smaller monitor region for screen capture
            monitor = sct.monitors[1]
            crop_monitor = {
                "top": monitor["top"] + crop_y,
                "left": monitor["left"] + crop_x,
                "width": crop_w,
                "height": crop_h
            }

            while not quit_flag:
                detection_counter += 1

                # PERFORMANCE OPTIMIZATION: Send frame to YOLO thread for processing
                if detection_counter % detection_interval == 0:
                    # Capture only the cropped region instead of full screen
                    screen_crop_raw = np.array(sct.grab(crop_monitor))
                    screen_crop = cv2.cvtColor(screen_crop_raw, cv2.COLOR_BGRA2BGR)

                    # Send frame to YOLO thread (non-blocking)
                    yolo_thread.update_frame(screen_crop)

                # Get latest detection from YOLO thread (non-blocking)
                detection = yolo_thread.get_latest_detection()

                if detection.get("found"):
                    # Keep coordinates in crop space since output matches crop size
                    # No need to adjust coordinates - they're already in the right coordinate system
                    detection_buffer.append(detection)

                # PERFORMANCE OPTIMIZATION: Only copy background when needed
                output_frame = green_background.copy()

                webcam_frame, face_rgba = webcam_thread.get_frames()
                if detection.get("found") and face_rgba is not None:
                    smoothed = average_bbox(detection_buffer) if detection_buffer else detection

                    if smoothed:
                        x1, y1 = smoothed["top_left"]
                        x2, y2 = smoothed["bottom_right"]
                        w, h = x2 - x1, y2 - y1

                        # FIX: Scale face to fit properly in 512x512 output
                        if w > 0 and h > 0:
                            face_h, face_w = face_rgba.shape[:2]

                            # CRITICAL FIX: Apply additional scaling for 512x512 output
                            # The face from 1280x720 webcam is too big for 512x512 output
                            output_scale_factor = 0.3  # Scale down face significantly for 512x512

                            # Calculate target size based on detected region AND output constraints
                            target_w = min(w, int(face_w * output_scale_factor))
                            target_h = min(h, int(face_h * output_scale_factor))

                            # Ensure face doesn't exceed output bounds
                            max_w = output_width - x1
                            max_h = output_height - y1
                            target_w = min(target_w, max_w)
                            target_h = min(target_h, max_h)

                            if target_w > 0 and target_h > 0:
                                # Resize face to target size
                                face_resized = cv2.resize(face_rgba, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

                                # Position face in detected region (top-left aligned for now)
                                final_x = max(0, min(x1, output_width - target_w))
                                final_y = max(0, min(y1, output_height - target_h))

                                output_frame = overlay_rgba(output_frame, face_resized, (final_x, final_y))

                        if debug_mode:
                            cv2.circle(output_frame, smoothed["center"], 3, (0, 255, 255), -1)
                            cv2.putText(output_frame, "FOUND", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                            # Show detection box
                            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                elif debug_mode:
                    cv2.putText(output_frame, "NOT FOUND", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

                frame_count += 1
                now = time.time()
                elapsed = now - prev_time
                if elapsed >= 1.0:
                    fps = frame_count / elapsed
                    frame_count = 0
                    prev_time = now

                if debug_mode:
                    # Show multiple debug windows to diagnose the issue

                    # 1. Screen crop (game region)
                    if detection_counter % detection_interval == 0:
                        screen_crop_raw = np.array(sct.grab(crop_monitor))
                        screen_crop = cv2.cvtColor(screen_crop_raw, cv2.COLOR_BGRA2BGR)
                        preview = cv2.resize(screen_crop, (400, 400))
                        cv2.imshow("1. Screen Crop", preview)

                    # 2. Raw camera feed
                    if webcam_frame is not None:
                        camera_preview = cv2.resize(webcam_frame, (400, 300))
                        cv2.imshow("2. Raw Camera Feed", camera_preview)

                    # 3. Segmented face (if available)
                    if face_rgba is not None:
                        face_preview = cv2.resize(face_rgba, (200, 200))
                        # Convert RGBA to BGR for display
                        face_bgr = cv2.cvtColor(face_preview, cv2.COLOR_RGBA2BGR)
                        cv2.imshow("3. Segmented Face", face_bgr)

                    # 4. Final virtual camera output
                    output_preview = cv2.resize(output_frame, (400, 400))
                    cv2.imshow("4. Virtual Camera Output", output_preview)

                    # Adjust text position for smaller output
                    cv2.putText(output_frame, f"FPS: {fps:.1f}", (10, output_height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:
                        debug_mode = False
                        cv2.destroyAllWindows()
                    elif key == ord('w'):
                        crop_y = max(0, crop_y - 10)
                        crop_monitor["top"] = monitor["top"] + crop_y
                    elif key == ord('s'):
                        crop_y += 10
                        crop_monitor["top"] = monitor["top"] + crop_y
                    elif key == ord('a'):
                        crop_x = max(0, crop_x - 10)
                        crop_monitor["left"] = monitor["left"] + crop_x
                    elif key == ord('f'):
                        crop_x += 10
                        crop_monitor["left"] = monitor["left"] + crop_x

                cam.send(cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB))
                cam.sleep_until_next_frame()

    # PERFORMANCE OPTIMIZATION: Properly cleanup threaded components
    webcam_thread.stop()
    yolo_thread.stop()
    webcam_thread.join()
    yolo_thread.join()
    segmentor.release()
    cv2.destroyAllWindows()
    listener.stop()

if __name__ == "__main__":
    main()
