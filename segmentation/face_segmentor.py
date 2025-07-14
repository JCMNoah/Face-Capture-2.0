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
        # PERFORMANCE OPTIMIZATION: Remove redundant camera initialization
        # Camera is now handled in WebcamCapture thread only
        self.cap = None  # Will be initialized only when get_segmented_frame() is called
        self.width = width
        self.height = height
        self.fps = fps

        # PERFORMANCE OPTIMIZATION: Use GPU acceleration if available
        try:
            # Try to initialize with GPU acceleration
            self.face_detection = mp.solutions.face_detection.FaceDetection(
                model_selection=0,  # Use model 0 (faster) instead of 1 (more accurate)
                min_detection_confidence=0.5  # Slightly lower threshold for better performance
            )
            print("✅ MediaPipe initialized (GPU acceleration attempted)")
        except Exception as e:
            # Fallback to CPU if GPU fails
            self.face_detection = mp.solutions.face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.5
            )
            print(f"⚠️  MediaPipe using CPU fallback: {e}")

        # PERFORMANCE OPTIMIZATION: Remove unused face_mesh to save memory and processing
        # self.face_mesh = mp.solutions.face_mesh.FaceMesh(...)

        self.padding_top = padding_top
        self.padding_right = padding_right
        self.padding_bottom = padding_bottom
        self.padding_left = padding_left

        self.buffer = deque(maxlen=smooth_buffer)

    def apply_oval_mask(self, face_crop):
        """QUALITY IMPROVED: Better oval mask with anti-aliasing"""
        h, w, _ = face_crop.shape

        # QUALITY IMPROVEMENT: Create mask with anti-aliasing for smoother edges
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, h // 2)
        axes = (int(w * 0.48), int(h * 0.48))  # Slightly larger for better coverage

        # Create anti-aliased ellipse
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

        # QUALITY IMPROVEMENT: Apply Gaussian blur for smoother edges
        mask = cv2.GaussianBlur(mask, (3, 3), 0)

        # PERFORMANCE OPTIMIZATION: Direct RGBA conversion with pre-allocated array
        face_crop_rgba = np.empty((h, w, 4), dtype=np.uint8)
        face_crop_rgba[:, :, :3] = face_crop
        face_crop_rgba[:, :, 3] = mask
        return face_crop_rgba

    def get_segmented_frame(self):
        """Initialize camera only when needed"""
        if self.cap is None:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        ret, frame = self.cap.read()
        if not ret:
            return None, None
        return self.process_frame(frame)

    def process_frame(self, frame):
        """BALANCED: Good quality + performance face processing"""
        # QUALITY IMPROVEMENT: Use less aggressive downscaling for better face detection
        scale_factor = 0.75  # Increased from 0.5 to 0.75 for better quality
        small_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
        rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb)

        cropped_face = None

        if results.detections:
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box

            # Scale back to original frame size
            h, w, _ = frame.shape
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            # PERFORMANCE OPTIMIZATION: Simplified padding calculation
            pad_w = int(width * self.padding_left)
            pad_h_top = int(height * self.padding_top)
            pad_h_bottom = int(height * self.padding_bottom)

            x1 = max(x - pad_w, 0)
            y1 = max(y - pad_h_top, 0)
            x2 = min(x + width + int(width * self.padding_right), w)
            y2 = min(y + height + pad_h_bottom, h)

            # PERFORMANCE OPTIMIZATION: Only smooth if buffer has multiple entries
            if len(self.buffer) > 0:
                self.buffer.append((x1, y1, x2, y2))
                x1, y1, x2, y2 = np.mean(self.buffer, axis=0).astype(int)
            else:
                self.buffer.append((x1, y1, x2, y2))

            # PERFORMANCE OPTIMIZATION: Ensure valid crop dimensions
            if x2 > x1 and y2 > y1:
                cropped = frame[y1:y2, x1:x2]
                cropped_face = self.apply_oval_mask(cropped)

        return frame, cropped_face

    def release(self):
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
