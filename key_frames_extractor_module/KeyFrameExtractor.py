from ..hand_detection_module.HandDetector import HandDetector
import cv2
import os

class KeyFrameExtractor:
    def __init__(self) -> None:
        self.hand_detector = HandDetector(is_image=True, num_hands=2, detection_confidence=0.5, tracking_confidence=0.5)
    
    def calcular_diferencia(self, puntos_previos, puntos_actuales):
        if puntos_previos is None or puntos_actuales is None:
            return float(1.0)
        
        # Comprobamos si alguna de las listas está vacía o si las dos listas tienen diferentes longitudes
        if len(puntos_previos) != len(puntos_actuales):
            return float(1.0)

        num_hands_previas = len(puntos_previos)
        print(num_hands_previas)
        total_diff = 0.0

        for i in range(num_hands_previas):
            hand_prev = puntos_previos[i].landmark
            hand_actual = puntos_actuales[i].landmark

            diff = sum([
                abs(lm1.x - lm2.x) + abs(lm1.y - lm2.y) + abs(lm1.z - lm2.z)
                for lm1, lm2 in zip(hand_prev, hand_actual)
            ])
            
            total_diff += diff

        if num_hands_previas >= 2:
            total_diff /= num_hands_previas
        return float(total_diff)
    
    def extract_key_frames(self, video_path, threshold=0.4, min_frame_interval=6):
        cap = cv2.VideoCapture(video_path)
        if not os.path.isfile(video_path):
            print(f"Error: El archivo {video_path} no existe.")
        if not cap.isOpened():
            print("Error: No se pudo abrir el video.")
            return []
        
        puntos_previos = []
        key_frames = []
        frame_count = 0
        print("Iniciando la extracción de frames clave")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Fin del video o error al leer frame")
                break
            
            frame_count += 1
            results = self.hand_detector.extractPoints(frame)
            
            if results.multi_hand_landmarks:
                puntos_actuales = results.multi_hand_landmarks
                diff = self.calcular_diferencia(puntos_previos, puntos_actuales)

                if diff > threshold and (len(key_frames) == 0 or frame_count - key_frames[-1][1] > min_frame_interval):
                    key_frames.append((frame, frame_count))
                    puntos_previos = puntos_actuales
            else:
                puntos_previos = None

        cap.release()
        return [frame for frame, _ in key_frames]