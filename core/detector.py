import cv2

class YuNetDetector:
    def __init__(self, model_path):
        self.model_path = model_path
        self.detector = cv2.FaceDetectorYN.create(model_path, "", (320, 320))
    
    def detect_faces(self, frame):
        h, w, _ = frame.shape
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        return faces