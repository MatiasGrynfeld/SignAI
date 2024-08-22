
from colorama import Fore, Style
from PoseDetectorClass import PoseDetector
import numpy as np
class KeyFrameExtractor:
    def __init__(self) -> None:
        self.poseDetector = PoseDetector(is_image=True, num_hands=2, detection_confidence=0.5, tracking_confidence=0.5)
    
    def calcularDiferencia(self, puntos_previos, puntos_actuales, handedness_prev, handedness_actual):
        if puntos_previos is None or puntos_actuales is None:
            return 10.0
        if len(puntos_previos) != len(puntos_actuales):
            return 10.0

        num_hands_previas = len(puntos_previos)
        num_hands_actuales = len(puntos_actuales)

        if num_hands_previas != num_hands_actuales:
            return 10.0

        total_diff = 0.0
        if len(puntos_previos) == len(puntos_actuales) and len(puntos_actuales) == 2:
            if handedness_prev[0].classification[0].label != handedness_actual[0].classification[0].label:
                puntos_actuales = puntos_actuales[::-1]
        for i in range(num_hands_previas):
            hand_prev = puntos_previos[i].landmark
            hand_actual = puntos_actuales[i].landmark

            diff = sum([
                abs(lm1.x - lm2.x) + abs(lm1.y - lm2.y) + abs(lm1.z - lm2.z)
                for lm1, lm2 in zip(hand_prev, hand_actual)
            ])
            total_diff += diff
        total_diff/=21
        if num_hands_previas >= 2:
            total_diff /= num_hands_previas
        return float(total_diff)
    
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
        handedness_prev = []
        key_frames = []
        frame_count = 0
        differencia=[]
        thresholdadjusted=[]

        for frame in self.extractFrames(video):
            frame_count += 1

            if len(key_frames)==0:
                results = self.poseDetector.extractPoints(frame)
                if results[0].multi_hand_landmarks:
                    handedness_prev = results[0].multi_handedness if results[0].multi_handedness else []
                    puntos_previos=results[0].multi_hand_landmarks
                    threshold = self.adjust_threshold(puntos_previos)
                    if return_frame:
                        if draw:
                            frame = self.poseDetector.drawLandmarks(results, frame)
                        key_frames.append((frame, frame_count))
                    else:
                        key_frames.append((results, frame_count))

            elif frame_count - key_frames[-1][1] > min_frame_interval:
                results = self.poseDetector.extractPoints(frame)
                if results[0].multi_hand_landmarks:
                    puntos_actuales = results[0].multi_hand_landmarks
                    handedness_actual = results[0].multi_handedness if results[0].multi_handedness else []
                    diff = self.calcularDiferencia(puntos_previos, puntos_actuales, handedness_prev, handedness_actual)
                    differencia.append(diff)
                    th=self.adjust_threshold(puntos_actuales)
                    thresholdadjusted.append((diff, th))
                    if diff > threshold:
                        if return_frame:
                            if draw:
                                frame = self.poseDetector.drawLandmarks(results, frame)
                            key_frames.append((frame, frame_count))
                        else:
                            key_frames.append((results, frame_count))
                        puntos_previos = puntos_actuales
                        handedness_prev = handedness_actual
                else:
                    puntos_previos = None
        for a in thresholdadjusted:
            print(f"{Fore.GREEN}{a}{Style.RESET_ALL}\n")
        return [frame for frame, _ in key_frames]
    
    def adjust_threshold(self, puntos_actuales, min_threshold=0.075, max_threshold=0.4):
        hands_size=[]
        # z_values=[]
        # for handlandmarks in puntos_actuales.multi_hand_world_landmarks:
        #     z_values.extend(landmark.z for landmark in handlandmarks.landmark)
        # z_prom=sum(z_values)/len(z_values)
        # threshold= base_threshold+base_threshold/z_prom
        for hand_landmarks in puntos_actuales:
            x_coords = [landmark.x for landmark in hand_landmarks.landmark]
            y_coords = [landmark.y for landmark in hand_landmarks.landmark]
            hand_width = max(x_coords) - min(x_coords)
            hand_height = max(y_coords) - min(y_coords)
            hands_size.append(hand_width * hand_height)
        if hands_size:
            hands_media_size=sum(hands_size)/len(puntos_actuales)
        mid_size = 0.25  # Puedes ajustar este valor
        k = 7.5  # Ajusta para cambiar la sensibilidad
        threshold = min_threshold + (max_threshold - min_threshold) / (1 + np.exp(-k * (hands_media_size - mid_size)))
        print(Fore.GREEN+f'Threshold: ${threshold} \n Hands size: ${hands_media_size}'+Style.RESET_ALL)
        return threshold