# import cv2
# import mediapipe as mp
# from forAllTest import init

# init()
# from FrameExtractorClass import FrameExtractor
# from pointsDetectorClass import pointsDetector

# # Inicializar MediaPipe Holistic
# mp_holistic = mp.solutions.holistic

# # Ruta de la imagen
# image_path = 'C:\\Users\\48113164\\Documents\\GitHub\\SignAI-IA.dev\\AI-ML_Development\\recursos\\a.jpg'

# # Leer la imagen
# frame = cv2.imread(image_path)
# if frame is None:
#     raise ValueError("Error al leer la imagen. Verifique la ruta.")

# is_image = True
# detection_confidence = 0.5
# tracking_confidence = 0.5

# with mp_holistic.Holistic(
#     static_image_mode=is_image,
#     min_detection_confidence=detection_confidence,
#     min_tracking_confidence=tracking_confidence
# ) as holistic:
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = holistic.process(frame_rgb)

#     if results.right_hand_landmarks:
#         # Imprimir la posición Z del primer punto de referencia (landmark)
#         print(results.right_hand_landmarks.landmark[0].z)
#     else:
#         print("No se detectaron landmarks de la mano derecha.")

import cv2
from forAllTest import init
import mediapipe as mp

init()
from FrameExtractorClass import FrameExtractor
from pointsDetectorClass import pointsDetector

video = cv2.VideoCapture('C:\\Users\\48113164\\Documents\\GitHub\\SignAI-IA.dev\\AI-ML_Development\\recursos\\videoprueba2.mp4')

FE = FrameExtractor()
Frames = FE.extractKeyFrames(video)
print(len(Frames))
for frame in Frames:
    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()