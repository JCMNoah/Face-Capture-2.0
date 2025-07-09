from segmentation.face_segmentor import FaceSegmentor
import cv2
import numpy as np

def display_rgba_on_black(rgba_img):
    # Blend RGBA onto black background
    bgr = cv2.cvtColor(rgba_img, cv2.COLOR_BGRA2BGR)
    alpha = rgba_img[:, :, 3] / 255.0
    background = np.zeros_like(bgr)
    blended = (bgr * alpha[..., None] + background * (1 - alpha[..., None])).astype(np.uint8)
    return blended

def main():
    segmentor = FaceSegmentor()

    while True:
        _, cropped = segmentor.get_segmented_frame()
        if cropped is None:
            continue

        display = display_rgba_on_black(cropped)
        cv2.imshow("Oval Face Only", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    segmentor.release()

if __name__ == "__main__":
    main()
