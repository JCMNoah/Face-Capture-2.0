import cv2
import mediapipe as mp
import numpy as np

class FaceSegmentor:
    def __init__(self, model_selection=1, width=1280, height=720, fps=60):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        self.segmentor = mp.solutions.selfie_segmentation.SelfieSegmentation(
            model_selection=model_selection
        )

    def get_segmented_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.segmentor.process(rgb)
        mask = results.segmentation_mask

        condition = mask > 0.6
        bg = np.zeros_like(frame)
        output = np.where(condition[..., None], frame, bg)

        return frame, output

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
