import cv2
import mediapipe as mp

class PoseDetector:
    def __init__(self, is_image, num_hands, detection_confidence, tracking_confidence) -> None:
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.is_image = is_image
        self.num_hands = num_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

    def extractPoints(self, frame):
        with self.mp_hands.Hands(
            static_image_mode=self.is_image,
            min_detection_confidence=self.detection_confidence
        ) as hands, self.mp_pose.Pose(
            static_image_mode=self.is_image,
            min_detection_confidence=self.detection_confidence
        ) as pose:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resultsHands = hands.process(frame_rgb)
            resultsPose = pose.process(frame_rgb)
            results = [resultsHands, resultsPose]
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
        
        self.pose_style = (
            self.mp_drawing.DrawingSpec(color=(255, 50, 0), thickness=4, circle_radius=5),
            self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=4, circle_radius=5)
        )
        
        if results[0].multi_hand_landmarks:
            for hand_landmarks in results[0].multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=4, circle_radius=5),
                    self.mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=4)
                )

        
        # if results.face_landmarks:
        #     self.mp_drawing.draw_landmarks(
        #         frame, results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION, self.face_style[0], self.face_style[1]
        #     )
        
        if results[1].pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results[1].pose_landmarks, self.mp_pose.POSE_CONNECTIONS, self.pose_style[0], self.pose_style[1]
            )
        
        return frame