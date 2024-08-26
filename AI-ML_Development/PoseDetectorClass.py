import cv2
import mediapipe as mp

class PoseDetector:
    def __init__(self, is_image, num_hands, detection_confidence, tracking_confidence) -> None:
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic
        self.mp_face_mesh = mp.solutions.face_mesh
        self.is_image = is_image
        self.num_hands = num_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

    def extractPoints(self, frame):
        with self.mp_holistic.Holistic(
            static_image_mode=self.is_image,
            min_detection_confidence=self.detection_confidence,
            model_complexity=2
        ) as holistic:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)
            return results
    
    def drawLandmarks(self, results, frame):
        if results is None:
            return frame

        self.face_style = (
            self.mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            self.mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
        )
        self.style = (
            self.mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=4, circle_radius=5),
            self.mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=4, circle_radius=5)
        )
        
        self.pose_style = (
            self.mp_drawing.DrawingSpec(color=(255, 50, 0), thickness=4, circle_radius=5),
            self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=4, circle_radius=5)
        )
        
        self.mp_drawing.draw_landmarks(
            frame, results.face_landmarks, self.mp_face_mesh.FACEMESH_CONTOURS, self.face_style[0], self.face_style[1])
        
        self.mp_drawing.draw_landmarks(
            frame, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS, self.style[0], self.style[1])
        
        self.mp_drawing.draw_landmarks(
            frame, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS, self.style[0], self.style[1])
        
        self.mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS, self.pose_style[0], self.pose_style[1])
        return frame


if __name__ == '__main__':
    image = cv2.imread("C:\\Users\\48519558\\Desktop\\SignAI-ML\\AI-ML_Development\\Resources\\a.jpg")
    pd = PoseDetector(is_image=True, num_hands=2, detection_confidence=0.75, tracking_confidence=0.75)
    r = pd.extractPoints(frame=image)
    new_image = pd.drawLandmarks(results=r, frame=image)
    cv2.imshow("Image", new_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    r_left_hand = r.left_hand_landmarks
    r_right_hand = r.right_hand_landmarks
    r_pose = r.pose_landmarks
    r_face = r.face_landmarks
    with open("output.txt", "w") as file:
        file.write(str(r_left_hand) + "\n")
        file.write("enter\n")
        file.write(str(r_right_hand) + "\n")
        file.write("enter\n")
        file.write(str(r_pose) + "\n")
        file.write("enter\n")
        file.write(str(r_face) + "\n")
