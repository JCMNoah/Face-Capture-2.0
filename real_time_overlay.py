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

    if y + h > base_img.shape[0] or x + w > base_img.shape[1]:
        return base_img

    overlay_rgb = overlay_img[:, :, :3]
    alpha_mask = overlay_img[:, :, 3] / 255.0

    base_region = base_img[y:y+h, x:x+w]
    blended = (overlay_rgb * alpha_mask[..., None] + base_region * (1 - alpha_mask[..., None])).astype(np.uint8)
    base_img[y:y+h, x:x+w] = blended
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
        self.segmentation_interval = 4
        self.counter = 0

    def run(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 60)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

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

    webcam_thread = WebcamCapture(segmentor)
    webcam_thread.start()

    detection_buffer = deque(maxlen=5)
    output_shape = (1080, 1920, 3)
    green_background = np.full(output_shape, (0, 255, 0), dtype=np.uint8)

    crop_x, crop_y, crop_w, crop_h = 700, 200, 512, 512

    with pyvirtualcam.Camera(width=1920, height=1080, fps=60) as cam:
        print(f"OBS virtual camera started at 1920x1080 @ 60fps")
        prev_time = time.time()
        frame_count = 0
        fps = 0

        with mss.mss() as sct:
            monitor = sct.monitors[1]

            while not quit_flag:
                screen = np.array(sct.grab(monitor))
                screen_bgr = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)
                screen_crop = screen_bgr[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                detection = detector.detect(screen_crop)
                output_frame = green_background.copy()

                webcam_frame, face_rgba = webcam_thread.get_frames()
                if detection.get("found") and face_rgba is not None:
                    detection["top_left"] = (detection["top_left"][0] + crop_x, detection["top_left"][1] + crop_y)
                    detection["bottom_right"] = (detection["bottom_right"][0] + crop_x, detection["bottom_right"][1] + crop_y)
                    detection["center"] = (detection["center"][0] + crop_x, detection["center"][1] + crop_y)
                    detection_buffer.append(detection)
                    smoothed = average_bbox(detection_buffer)

                    if smoothed:
                        x1, y1 = smoothed["top_left"]
                        x2, y2 = smoothed["bottom_right"]
                        w, h = x2 - x1, y2 - y1
                        face_resized = cv2.resize(face_rgba, (w, h), interpolation=cv2.INTER_AREA)
                        output_frame = overlay_rgba(output_frame, face_resized, smoothed["top_left"])
                        if debug_mode:
                            cv2.circle(output_frame, smoothed["center"], 4, (0, 255, 255), -1)
                            cv2.putText(output_frame, "FOUND", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif debug_mode:
                    cv2.putText(output_frame, "NOT FOUND", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                frame_count += 1
                now = time.time()
                elapsed = now - prev_time
                if elapsed >= 1.0:
                    fps = frame_count / elapsed
                    frame_count = 0
                    prev_time = now

                if debug_mode:
                    preview = cv2.resize(screen_crop, (500, 500))
                    cv2.imshow("Screen Crop", preview)
                    cv2.putText(output_frame, f"FPS: {fps:.1f}", (10, output_shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:
                        debug_mode = False
                        cv2.destroyWindow("Screen Crop")
                    elif key == ord('w'): crop_y = max(0, crop_y - 10)
                    elif key == ord('s'): crop_y += 10
                    elif key == ord('a'): crop_x = max(0, crop_x - 10)
                    elif key == ord('f'): crop_x += 10

                cam.send(cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB))
                cam.sleep_until_next_frame()

    webcam_thread.stop()
    webcam_thread.join()
    segmentor.release()
    cv2.destroyAllWindows()
    listener.stop()

if __name__ == "__main__":
    main()
