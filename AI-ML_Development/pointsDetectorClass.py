import cv2
import mediapipe as mp
import os
import time

class pointsDetector:
    def __init__(self,is_image, num_hands, detection_confidence, tracking_confidence):
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_holistic = mp.solutions.holistic
            self.is_image = is_image
            self.num_hands = num_hands
            self.detection_confidence = detection_confidence
            self.tracking_confidence = tracking_confidence

    def extractPoints(self, frame, model):
        #frame = cv2.flip(frame, 1)
        with self.mp_holistic.Holistic(
            static_image_mode=self.is_image,
            max_num_hands=self.num_hands,
            min_detection_confidence=self.detection_confidence
        ) as holistic:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)
            return results

    def drawLandmarks(self, results, frame):
        if results.multi_landmarks:
            self.face_style=(self.mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1, circle_radius=3),
                                            self.mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1))
            self.hand_style=(self.mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=4, circle_radius=5),
                                            self.mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=4))
            self.pose_style=(self.mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=5, circle_radius=6),
                                            self.mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=6))
            self.mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS, self.hand_style)
            self.mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS, self.hand_style)
            self.mp_drawing.draw_landmarks(frame, results.face_landmarks, self.mp_holistic.FACE_CONNECTIONS, self.face_style)
            self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS, self.pose_style)
        return frame