import cv2
import numpy as np

class CharacterDetector:
    def __init__(self, template_path, match_threshold=0.7):
        self.template = cv2.imread(template_path, cv2.IMREAD_COLOR)
        self.template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        self.thresh = match_threshold
        self.w, self.h = self.template_gray.shape[::-1]

    def detect(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(frame_gray, self.template_gray, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if max_val >= self.thresh:
            top_left = max_loc
            bottom_right = (top_left[0] + self.w, top_left[1] + self.h)
            center = (top_left[0] + self.w // 2, top_left[1] + self.h // 2)
            return {
                "found": True,
                "top_left": top_left,
                "bottom_right": bottom_right,
                "center": center,
                "score": max_val
            }
        else:
            return {"found": False}
