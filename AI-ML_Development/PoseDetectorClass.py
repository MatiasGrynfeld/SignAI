import cv2
import mediapipe as mp

class PoseDetector:
    def __init__(self, is_image, detection_confidence, tracking_confidence) -> None:
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic
        self.is_image = is_image
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

    def extractPoints(self, frame):
        with self.mp_holistic.Holistic(
            static_image_mode=self.is_image,
            min_detection_confidence=self.detection_confidence
        ) as holistic:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)
            return results
    
    def drawLandmarks(self, results, frame):
        if results is None:
            return frame

        # self.face_style = (
        #     self.mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
        #     self.mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
        # )
        self.style = (
            self.mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=4, circle_radius=5),
            self.mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=4, circle_radius=5)
        )
        
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS, self.style[0], self.style[1]
            )
        
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS, self.style[0], self.style[1]
            )
        
        # if results.face_landmarks:
        #     self.mp_drawing.draw_landmarks(
        #         frame, results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION, self.face_style[0], self.face_style[1]
        #     )
        
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS, self.style[0], self.style[1]
            )
        
        return frame