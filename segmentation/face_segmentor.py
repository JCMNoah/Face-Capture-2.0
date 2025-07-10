import cv2
import mediapipe as mp
import numpy as np
from collections import deque

class FaceSegmentor:
    def __init__(
        self,
        model_selection=1,
        width=1280,
        height=720,
        fps=60,
        padding_top=0.4,
        padding_right=0.15,
        padding_bottom=0.05,
        padding_left=0.15,
        smooth_buffer=5
    ):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=0.6
        )

        self.padding_top = padding_top
        self.padding_right = padding_right
        self.padding_bottom = padding_bottom
        self.padding_left = padding_left

        self.buffer = deque(maxlen=smooth_buffer)

    def apply_oval_mask(self, face_crop):
        h, w, _ = face_crop.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, h // 2)
        axes = (int(w * 0.5), int(h * 0.5))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

        face_crop_rgba = cv2.cvtColor(face_crop, cv2.COLOR_BGR2BGRA)
        face_crop_rgba[:, :, 3] = mask  # apply mask to alpha channel
        return face_crop_rgba

    def get_segmented_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb)

        cropped_face = None

        if results.detections:
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            # Padding
            x1 = max(int(x - width * self.padding_left), 0)
            y1 = max(int(y - height * self.padding_top), 0)
            x2 = min(int(x + width + width * self.padding_right), w)
            y2 = min(int(y + height + height * self.padding_bottom), h)

            # Smoothing
            self.buffer.append((x1, y1, x2, y2))
            x1, y1, x2, y2 = np.mean(self.buffer, axis=0).astype(int)

            cropped = frame[y1:y2, x1:x2]
            cropped_face = self.apply_oval_mask(cropped)

        return frame, cropped_face

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
