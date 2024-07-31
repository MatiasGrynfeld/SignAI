import cv2
import mediapipe as mp

class HandDetector:
    def __init__(self, is_image, num_hands, detection_confidence, tracking_confidence) -> None:
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.is_image = is_image
        self.num_hands = num_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

    def extractPoints(self, frame):
        frame = cv2.flip(frame, 1)
        with self.mp_hands.Hands(
            static_image_mode=self.is_image,
            max_num_hands=self.num_hands,
            min_detection_confidence=self.detection_confidence
        ) as hands:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            return results

    def drawLandmarks(self, results, frame):
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=4, circle_radius=5),
                    self.mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=4)
                )
        return frame
