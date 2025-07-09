import cv2
from detection.character_detector import CharacterDetector
from pathlib import Path

base_dir = Path(__file__).parent

# Update paths as needed
template_path = base_dir / "data" / "template_face.png"  # Replace with actual path to face_17.png or similar
test_image_path = base_dir / "data" / "template_screenshot.png"  # Replace with actual path to full screenshot

detector = CharacterDetector(template_path)

img = cv2.imread(test_image_path)
result = detector.detect(img)

if result["found"]:
    cv2.rectangle(img, result["top_left"], result["bottom_right"], (0, 255, 0), 2)
    cv2.circle(img, result["center"], 5, (0, 0, 255), -1)
    cv2.putText(img, f"Score: {result['score']:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
else:
    cv2.putText(img, "Not Found", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

cv2.imshow("Detection Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
