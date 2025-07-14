from ultralytics import YOLO
import cv2

class YOLODetector:
    def __init__(self, model_path='runs/detect/train7/weights/best.pt', conf_threshold=0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame):
        # Store original dimensions
        original_h, original_w = frame.shape[:2]

        # Run inference on original frame for accurate coordinates
        results = self.model(frame, verbose=False)[0]

        detections = []
        for box in results.boxes:
            conf = float(box.conf)
            if conf >= self.conf_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                detections.append({
                    'top_left': (x1, y1),
                    'bottom_right': (x2, y2),
                    'center': center,
                    'confidence': conf
                })

        # Return the first (highest confidence) detection, or empty
        if detections:
            detections[0]['found'] = True
            return detections[0]
        else:
            return {'found': False}

