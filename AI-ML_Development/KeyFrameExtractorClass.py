
from colorama import Fore, Style
from PoseDetectorClass import PoseDetector
import numpy as np
class KeyFrameExtractor:
    def __init__(self) -> None:
        self.poseDetector = PoseDetector(is_image=True, num_hands=2, detection_confidence=0.5, tracking_confidence=0.5)
    
    def calcularDiferencia(self, puntos_previos, puntos_actuales, hands_detected_prev, hands_detected_actual):
        diff=0.0
        if puntos_previos is None:
            return 1.0
        if puntos_actuales is None:
            return 0.0
        if hands_detected_prev!=hands_detected_actual:
            return 1.0
        if hands_detected_actual==2:
            right_hand_prev = puntos_previos.right_hand_landmarks
            right_hand_actual = puntos_actuales.right_hand_landmarks
            left_hand_prev=puntos_previos.left_hand_landmarks
            left_hand_actual = puntos_actuales.left_hand_landmarks
            diff+=self.calcularDifManos(right_hand_prev,right_hand_actual)
            diff+=self.calcularDifManos(left_hand_prev,left_hand_actual)

        elif hands_detected_actual==1:
            if left_hand_prev and left_hand_actual:
                total_diff = self.calcularDifManos(left_hand_prev, left_hand_actual)

            elif right_hand_prev and right_hand_actual:
                total_diff = self.calcularDifManos(right_hand_prev, right_hand_actual)

            else:
                return 1.0

        return float(diff)

    def calcularDifManos(self,hand_prev, hand_actual ):
        return sum([
            abs(lm1.x - lm2.x) + abs(lm1.y - lm2.y) + abs(lm1.z - lm2.z)
            for lm1, lm2 in zip(hand_prev.landmark, hand_actual.landmark)
        ])

    def extractFrames(self, video):
        ret = 1
        while ret:
            ret, frame = video.read()
            if not ret:
                break
            yield frame
        video.release()

    def extractKeyFrames(self, return_frame, draw, video, min_frame_interval=1):
        puntos_previos = []
        key_frames = []
        frame_count = 0
        differencia=[]
        thresholdadjusted=[]
        num_hands=0
        adjust_th=False

        for frame in self.extractFrames(video):
            frame_count += 1
            if frame_count %50 ==0:
                adjust_th=True
            if len(key_frames)==0:
                results = self.poseDetector.extractPoints(frame)
                num_hands_prev=0
                if results.right_hand_landmarks or results.left_hand_landmarks:
                    puntos_previos=results
                    if results.right_hand_landmarks:
                        num_hands_prev+=1
                    if results.left_hand_landmarks:
                        num_hands_prev+=1

                    threshold = self.adjust_threshold(puntos_previos,num_hands_prev)

                    if return_frame:
                        if draw:
                            frame = self.poseDetector.drawLandmarks(results, frame)
                        key_frames.append((frame, frame_count))
                    else:
                        key_frames.append((results, frame_count))

            elif frame_count - key_frames[-1][1] > min_frame_interval:
                results = self.poseDetector.extractPoints(frame)
                if results.right_hand_landmarks or results.left_hand_landmarks:
                    puntos_actuales = results
                    if results.right_hand_landmarks:
                        num_hands+=1
                    if results.left_hand_landmarks:
                        num_hands+=1
                    
                    if adjust_th:
                        threshold=self.adjust_threshold(puntos_actuales, num_hands)
                        adjust_th=False
                    diff = self.calcularDiferencia(puntos_previos, puntos_actuales,hands_detected_prev=num_hands_prev, hands_detected_actual=num_hands)
                    differencia.append(diff)
                    if diff > threshold:
                        if return_frame:
                            if draw:
                                frame = self.poseDetector.drawLandmarks(results, frame)
                            key_frames.append((frame, frame_count))
                        else:
                            key_frames.append((results, frame_count))
                        puntos_previos = puntos_actuales
                        num_hands_prev=num_hands
                else:
                    puntos_previos = None
        for a in thresholdadjusted:
            print(f"{Fore.GREEN}{a}{Style.RESET_ALL}\n")
        return [frame for frame, _ in key_frames]
    
    def adjust_threshold(self, puntos_actuales, num_hands, min_threshold=-0.35, max_threshold=0.35):
        hands_size=0.0
        if puntos_actuales.right_hand_landmarks:
            x_coords=[landmark.x for landmark in puntos_actuales.right_hand_landmarks.landmark]
            y_coords=[landmark.y for landmark in puntos_actuales.right_hand_landmarks.landmark]
            hand_width = max(x_coords) - min(x_coords)
            hand_height = max(y_coords) - min(y_coords)
            hands_size+=(hand_width * hand_height)

        if puntos_actuales.left_hand_landmarks:
            x_coords=[landmark.x for landmark in puntos_actuales.left_hand_landmarks.landmark]
            y_coords=[landmark.y for landmark in puntos_actuales.left_hand_landmarks.landmark]
            hand_width = max(x_coords) - min(x_coords)
            hand_height = max(y_coords) - min(y_coords)
            hands_size+=(hand_width * hand_height)
            
        if hands_size:
            hands_media_size=hands_size/num_hands
        mid_size = 0  # Puedes ajustar este valor
        k = 17  # Ajusta para cambiar la sensibilidad
        threshold = min_threshold + (max_threshold - min_threshold) / (1 + np.exp(-k * (hands_media_size - mid_size)))+0.03
        print(Fore.GREEN+f'Threshold: ${threshold} \n Hands size: ${hands_media_size}'+Style.RESET_ALL)
        return threshold