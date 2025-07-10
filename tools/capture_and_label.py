import cv2
import mss
import os
import numpy as np
from pathlib import Path

output_dir = Path("yolo_dataset")
img_dir = output_dir / "images/train"
label_dir = output_dir / "labels/train"
img_dir.mkdir(parents=True, exist_ok=True)
label_dir.mkdir(parents=True, exist_ok=True)

def capture_screen():
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        img = sct.grab(monitor)
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_BGRA2BGR)
        return img_bgr, monitor['width'], monitor['height']

class Annotator:
    def __init__(self, preview_size=(960, 540)):
        self.preview_size = preview_size
        self.image = None
        self.preview = None
        self.clone = None
        self.boxes = []
        self.scale_x = 1
        self.scale_y = 1
        self.drawing = False
        self.ix = self.iy = -1

    def set_image(self, image):
        self.image = image
        h, w = image.shape[:2]
        pw, ph = self.preview_size
        self.scale_x = w / pw
        self.scale_y = h / ph
        self.preview = cv2.resize(image, self.preview_size)
        self.clone = self.preview.copy()
        self.boxes = []

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.clone = self.preview.copy()
            for bx in self.boxes:
                cv2.rectangle(self.clone, bx[0], bx[1], (0, 255, 0), 2)
            cv2.rectangle(self.clone, (self.ix, self.iy), (x, y), (0, 255, 255), 2)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            x1 = int(min(self.ix, x))
            y1 = int(min(self.iy, y))
            x2 = int(max(self.ix, x))
            y2 = int(max(self.iy, y))
            self.boxes.append(((x1, y1), (x2, y2)))

    def undo(self):
        if self.boxes:
            self.boxes.pop()
            self.clone = self.preview.copy()
            for bx in self.boxes:
                cv2.rectangle(self.clone, bx[0], bx[1], (0, 255, 0), 2)

    def display(self, label="Label"):
        cv2.imshow(label, self.clone)

    def save(self, image_path, label_path, full_w, full_h):
        cv2.imwrite(str(image_path), self.image)
        with open(label_path, "w") as f:
            for (x1, y1), (x2, y2) in self.boxes:
                # Rescale to full res
                x1 = int(x1 * self.scale_x)
                x2 = int(x2 * self.scale_x)
                y1 = int(y1 * self.scale_y)
                y2 = int(y2 * self.scale_y)
                cx = ((x1 + x2) / 2) / full_w
                cy = ((y1 + y2) / 2) / full_h
                w = (x2 - x1) / full_w
                h = (y2 - y1) / full_h
                f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

def main():
    index = len(list(img_dir.glob("*.jpg"))) + 1
    annotator = Annotator(preview_size=(960, 540))
    cv2.namedWindow("Label", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Label", annotator.mouse_callback)

    print("🎮 Press 'n' to capture screen, 'z' to undo, ENTER to save, 'q' to quit")

    while True:
        if annotator.clone is not None:
            annotator.display()

        key = cv2.waitKey(1) & 0xFF

        if key == ord('n'):
            image, w, h = capture_screen()
            annotator.set_image(image)
            print(f"🖼 New screenshot loaded ({w}x{h}) — draw boxes")

        elif key == ord('z'):
            annotator.undo()
            print("↩️  Undid last box")

        elif key == 13:  # Enter
            if annotator.boxes:
                img_path = img_dir / f"snap_{index:03}.jpg"
                label_path = label_dir / f"snap_{index:03}.txt"
                annotator.save(img_path, label_path, w, h)
                print(f"✅ Saved: {img_path.name} with {len(annotator.boxes)} boxes")
                index += 1
            else:
                print("⚠️  No boxes to save")

        elif key == 27:  # ESC
            annotator.boxes = []
            annotator.clone = annotator.preview.copy()
            print("⛔ Cleared boxes for current image")

        elif key == ord('q'):
            print("👋 Exiting...")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
