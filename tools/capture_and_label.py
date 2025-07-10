from pathlib import Path
import cv2
import mss
import numpy as np
from ultralytics import YOLO

# === Directories ===
output_dir = Path("yolo_dataset")
img_dir = output_dir / "images/train"
label_dir = output_dir / "labels/train"
img_dir.mkdir(parents=True, exist_ok=True)
label_dir.mkdir(parents=True, exist_ok=True)

model = YOLO("yolov8n.pt")
history = []

# === Screen Capture ===
def capture_screen():
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        img = sct.grab(monitor)
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_BGRA2BGR)
        return img_bgr, monitor['width'], monitor['height']

# === Annotator Class with Box Resize ===
class Annotator:
    def __init__(self, preview_size=(960, 540)):
        self.preview_size = preview_size
        self.image = None
        self.preview = None
        self.clone = None
        self.box = None  # Only one box
        self.scale_x = 1
        self.scale_y = 1
        self.dragging = False
        self.drag_corner = None
        self.box_color = (0, 255, 255)
        self.resize_margin = 10

    def set_image(self, image):
        self.image = image
        h, w = image.shape[:2]
        pw, ph = self.preview_size
        self.scale_x = w / pw
        self.scale_y = h / ph
        self.preview = cv2.resize(image, self.preview_size)
        self.clone = self.preview.copy()
        self.box = None

    def in_resize_zone(self, x, y):
        if not self.box:
            return None
        (x1, y1), (x2, y2) = self.box
        corners = {
            'tl': (x1, y1),
            'tr': (x2, y1),
            'bl': (x1, y2),
            'br': (x2, y2)
        }
        for key, (cx, cy) in corners.items():
            if abs(x - cx) < self.resize_margin and abs(y - cy) < self.resize_margin:
                return key
        return None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            zone = self.in_resize_zone(x, y)
            if zone:
                self.dragging = True
                self.drag_corner = zone
            else:
                self.box = [(x, y), (x, y)]
                self.dragging = True
                self.drag_corner = 'br'

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            if self.box and self.drag_corner:
                x1, y1 = self.box[0]
                x2, y2 = self.box[1]
                if self.drag_corner == 'tl':
                    self.box = [(x, y), (x2, y2)]
                elif self.drag_corner == 'tr':
                    self.box = [(x1, y), (x, y2)]
                elif self.drag_corner == 'bl':
                    self.box = [(x, y1), (x2, y)]
                elif self.drag_corner == 'br':
                    self.box = [(x1, y1), (x, y)]
                self.update_preview()

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.drag_corner = None
            if self.box:
                x1, y1 = self.box[0]
                x2, y2 = self.box[1]
                self.box = [(min(x1, x2), min(y1, y2)), (max(x1, x2), max(y1, y2))]
                self.update_preview()

    def update_preview(self):
        self.clone = self.preview.copy()
        if self.box:
            cv2.rectangle(self.clone, self.box[0], self.box[1], self.box_color, 2)

    def undo(self):
        self.box = None
        self.update_preview()

    def display(self, label="Label"):
        cv2.imshow(label, self.clone)

    def save(self, image_path, label_path, full_w, full_h):
        if image_path.exists() or label_path.exists():
            print(f"❌ Skipping {image_path.name} (already exists)")
            return

        cv2.imwrite(str(image_path), self.image)
        with open(label_path, "w") as f:
            (x1, y1), (x2, y2) = self.box
            x1 = int(x1 * self.scale_x)
            x2 = int(x2 * self.scale_x)
            y1 = int(y1 * self.scale_y)
            y2 = int(y2 * self.scale_y)
            cx = ((x1 + x2) / 2) / full_w
            cy = ((y1 + y2) / 2) / full_h
            w = (x2 - x1) / full_w
            h = (y2 - y1) / full_h
            f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        history.append((image_path, label_path))
        print(f"✅ Saved: {image_path.name}")

# === Main Tool Loop ===
def run_auto_label():
    index = len(list(img_dir.glob("*.jpg"))) + 1
    annotator = Annotator()
    cv2.namedWindow("Label", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Label", annotator.mouse_callback)

    print("🎮 Press 'n' to capture, 'd' for auto-detect, 's' to save, 'u' to undo, 'q' to quit")

    while True:
        if annotator.clone is not None:
            annotator.display()

        key = cv2.waitKey(1) & 0xFF

        if key == ord('n'):
            image, w, h = capture_screen()
            annotator.set_image(image)
            print("🖼 Screenshot loaded")

        elif key == ord('d') and annotator.image is not None:
            results = model.predict(annotator.image, conf=0.5, imgsz=640, verbose=False)
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            if len(boxes):
                x1, y1, x2, y2 = boxes[0]
                y2 = y1 + int((y2 - y1) * 0.45)  # simulate tighter head-box
                # Convert to preview scale
                x1 = int(x1 / annotator.scale_x)
                x2 = int(x2 / annotator.scale_x)
                y1 = int(y1 / annotator.scale_y)
                y2 = int(y2 / annotator.scale_y)
                annotator.box = [(x1, y1), (x2, y2)]
                annotator.update_preview()
                print("🧠 Auto-detected head")

        elif key == ord('s'):
            if annotator.box:

                while True:
                    img_path = img_dir / f"snap_{index:03}.jpg"
                    label_path = label_dir / f"snap_{index:03}.txt"
                    if not img_path.exists() and not label_path.exists():
                        annotator.save(img_path, label_path, w, h)
                        index += 1
                        annotator.box = None
                        break
                    else:
                        index += 1

            else:
                print("⚠️ No box to save")

        elif key == ord('u'):
            if history:
                img_path, label_path = history.pop()
                if img_path.exists(): img_path.unlink()
                if label_path.exists(): label_path.unlink()
                print(f"🗑️ Undid {img_path.name}")
            else:
                print("Nothing to undo")
            annotator.undo()

        elif key == ord('q'):
            print("👋 Exiting...")
            break

    cv2.destroyAllWindows()

run_auto_label()

