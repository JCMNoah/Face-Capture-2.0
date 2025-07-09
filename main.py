from segmentation.face_segmentor import FaceSegmentor
import cv2

def main():
    segmentor = FaceSegmentor()

    while True:
        frame, segmented = segmentor.get_segmented_frame()
        if segmented is None:
            break

        cv2.imshow("Segmented Face", segmented)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    segmentor.release()

if __name__ == "__main__":
    main()
