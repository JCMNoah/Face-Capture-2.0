from pathlib import Path
import cv2
import numpy as np
import mss

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

def main():
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "runs/detect/train7/weights/best.pt"
    detector = YOLODetector(str(model_path), conf_threshold=0.5)

    segmentor = FaceSegmentor()
    face_rgba = None

    print("Opening webcam...")
    while True:
        full_frame, face = segmentor.get_segmented_frame()
        if full_frame is not None:
            cv2.imshow("Webcam Preview", full_frame)
        if face is not None:
            face_rgba = face
            print("Face captured.")
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Face capture aborted by user.")
            break

    segmentor.release()
    cv2.destroyAllWindows()

    if face_rgba is None:
        print("No face found after webcam preview.")
        return

    print("Starting live screen capture and overlay...")

    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Update if needed for multi-monitor setups

        while True:
            screen = np.array(sct.grab(monitor))
            screen_bgr = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)

            detection = detector.detect(screen_bgr)
            if detection["found"]:
                x1, y1 = detection["top_left"]
                x2, y2 = detection["bottom_right"]
                w, h = x2 - x1, y2 - y1

                face_resized = cv2.resize(face_rgba, (w, h), interpolation=cv2.INTER_AREA)
                screen_bgr = overlay_rgba(screen_bgr, face_resized, detection["top_left"])

                cv2.circle(screen_bgr, detection["center"], 4, (0, 255, 255), -1)
                cv2.putText(screen_bgr, "FOUND", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(screen_bgr, "NOT FOUND", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Live Overlay", screen_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
