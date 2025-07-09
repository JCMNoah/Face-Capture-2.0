import cv2
import numpy as np
from detection.character_detector import CharacterDetector
from segmentation.face_segmentor import FaceSegmentor
from pathlib import Path

def overlay_rgba(base_img, overlay_img, position):
    x, y = position
    h, w = overlay_img.shape[:2]

    # Bounds check
    if y + h > base_img.shape[0] or x + w > base_img.shape[1]:
        return base_img

    overlay_rgb = overlay_img[:, :, :3]
    alpha_mask = overlay_img[:, :, 3] / 255.0

    base_region = base_img[y:y+h, x:x+w]
    blended = (overlay_rgb * alpha_mask[..., None] + base_region * (1 - alpha_mask[..., None])).astype(np.uint8)
    base_img[y:y+h, x:x+w] = blended
    return base_img

def main():
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"

    screenshot_path = data_dir / "template_screenshot.png"
    template_path = data_dir / "template_face.png"

    # Load screenshot
    screenshot = cv2.imread(str(screenshot_path))
    if screenshot is None:
        raise FileNotFoundError("Missing screenshot at data/test_screenshot.png")

    # Setup character head detector and face segmentor
    detector = CharacterDetector(str(template_path))
    segmentor = FaceSegmentor(
        padding_top=0.4,
        padding_right=0.2,
        padding_bottom=0.5,
        padding_left=0.2
    )

    _, oval_face = segmentor.get_segmented_frame()
    if oval_face is None:
        print("No face detected.")
        return

    # Detect character head
    result = detector.detect(screenshot)
    if not result["found"]:
        print("Character head not found.")
        return

    # Resize oval face to match template dimensions
    face_resized = cv2.resize(oval_face, (detector.w, detector.h), interpolation=cv2.INTER_AREA)

    # Overlay onto screenshot
    composed = overlay_rgba(screenshot.copy(), face_resized, result["top_left"])

    cv2.imshow("Face Overlay", composed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    segmentor.release()

if __name__ == "__main__":
    main()
