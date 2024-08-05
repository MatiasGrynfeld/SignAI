from pointsDetectorClass import pointsDetector
from colorama import Fore, init
import cv2

init()
class FrameExtractor:
    def __init__(self) -> None:
        self.point_detector = pointsDetector(is_image=True, num_hands=2, detection_confidence=0.5, tracking_confidence=0.5)
        
    def extractFrames(self, video):
        ret = 1
        frames=[]
        while ret:
            ret, frame = video.read()
            if not ret:
                break
            frames.append(frame)
        for frame in frames:
            #threshold = self.adjust_threshold(puntos_previos)
            #print(Fore.GREEN + f"Threshold: {threshold}")
            #print(Fore.RESET)
            results = self.point_detector.extractPoints(frame)
            frame = self.point_detector.drawLandmarks(results, frame)
        video.release()
        return frames
               
    def adjust_threshold(self, puntos_actuales, base_threshold=40.0):
        for hand_landmarks in puntos_actuales:
            x_coords = [landmark.x for landmark in hand_landmarks.landmark]
            y_coords = [landmark.y for landmark in hand_landmarks.landmark]
        hand_width = max(x_coords) - min(x_coords)
        hand_height = max(y_coords) - min(y_coords)
        hands_size= hand_width * hand_height
        threshold=base_threshold*hands_size
        return threshold