from pointsDetectorClass import pointsDetector
from colorama import Fore, init
import cv2

init()
class FrameExtractor:
    def __init__(self) -> None:
        self.point_detector = pointsDetector(is_image=True, detection_confidence=0.5, tracking_confidence=0.5)
    
    def calcularDiferencia(self, puntos_previos, puntos_actuales, hands_detected_prev, hands_detected_actual):
        if puntos_previos is None:
            return 100.0
        if puntos_actuales is None:
            return 0.0
        if hands_detected_prev!=hands_detected_actual:
            return 100.0
        right_hand_prev = puntos_previos.right_hand_landmarks
        right_hand_actual = puntos_actuales.right_hand_landmarks
        left_hand_prev=puntos_previos.left_hand_landmarks
        left_hand_actual = puntos_actuales.left_hand_landmarks
        diff=self.calcularDifManos(right_hand_prev,right_hand_actual,left_hand_prev,left_hand_actual,hands_detected_actual)
        return float(diff)
    
    def calcularDifManos(self, right_hand_prev, right_hand_actual, left_hand_prev, left_hand_actual, num_hands):
        total_diff = 0.0
        
        def calc_diff(hand_prev, hand_actual):
            return sum([
                abs(lm1.x - lm2.x) + abs(lm1.y - lm2.y) + abs(lm1.z - lm2.z)
                for lm1, lm2 in zip(hand_prev.landmark, hand_actual.landmark)
            ])
        
        if num_hands == 2:
            if right_hand_prev and right_hand_actual:
                total_diff += calc_diff(right_hand_prev, right_hand_actual)
            if left_hand_prev and left_hand_actual:
                total_diff += calc_diff(left_hand_prev, left_hand_actual)
            total_diff /= 2
            
        elif num_hands == 1:
            if left_hand_prev and left_hand_actual:
                total_diff = calc_diff(left_hand_prev, left_hand_actual)
            elif right_hand_prev and right_hand_actual:
                total_diff = calc_diff(right_hand_prev, right_hand_actual)
            else:
                return 100.0
        
        return total_diff

    def extractKeyFrames(self, video, min_frame_interval=1):
        puntos_previos = []
        manos_detectadas_prev=0
        manos_detectadas_actual=0
        key_frames = []
        frame_count = 0
        ret = 1
        while ret:
            ret, frame = video.read()
            if not ret:
                break
            frame_count += 1
            if len(key_frames)==0:
                results = self.point_detector.extractPoints(frame)
                if results.right_hand_landmarks or results.left_hand_landmarks:
                    puntos_previos=results
                    threshold = self.adjust_threshold(puntos_previos)
                    print(Fore.GREEN + f"Threshold: {threshold}")
                    print(Fore.RESET)
                    frame = self.point_detector.drawLandmarks(puntos_previos, frame)
                    key_frames.append((frame, frame_count))
                    if results.right_hand_landmarks and results.left_hand_landmarks:
                        manos_detectadas_prev=2
                    else:
                        manos_detectadas_prev=1

            elif frame_count - key_frames[-1][1] > min_frame_interval:
                results = self.point_detector.extractPoints(frame)
                if results.right_hand_landmarks or results.left_hand_landmarks:
                    puntos_actuales = results
                    if results.right_hand_landmarks and results.left_hand_landmarks:
                        manos_detectadas_actual=2
                    else:
                        manos_detectadas_actual=1
                    diff = self.calcularDiferencia(puntos_previos, puntos_actuales, manos_detectadas_prev, manos_detectadas_actual)

                    if diff > threshold:
                        frame = self.point_detector.drawLandmarks(puntos_actuales, frame)
                        key_frames.append((frame, frame_count))
                        puntos_previos = puntos_actuales
                else:
                    puntos_previos = None
        video.release()
        return [frame for frame, _ in key_frames]
         
    def adjust_threshold(self, frame, base_threshold=4.0):
        results = self.point_detector.extractPoints(frame)
        y_values = []
        z_values=[]
        x_values=[]
        ysum=0
        xsum=0
        zsum=0
        if results.right_hand_landmarks.landmark:
            for z in results.right_hand_landmarks.landmark:
                ysum+= z.y
                xsum+=z.x
                zsum+=z.z
            y_values.append(ysum/len(results.right_hand_landmarks.landmark))
            x_values.append(xsum/len(results.right_hand_landmarks.landmark))
            z_values.append(zsum/len(results.right_hand_landmarks.landmark))
            
        if results.left_hand_landmarks:
            for z in results.left_hand_landmarks.landmark:
                ysum+= z.y
                xsum+=z.x
                zsum+=z.z
            y_values.append(ysum/len(results.left_hand_landmarks.landmark))
            x_values.append(xsum/len(results.left_hand_landmarks.landmark))
            z_values.append(zsum/len(results.left_hand_landmarks.landmark))
        
        if not y_values:
            return base_threshold
        
        average_y = sum(y_values) / len(y_values)
        average_x = sum(x_values) / len(x_values)
        average_z = sum(z_values) / len(z_values)
        adjusted_threshold = base_threshold / average_y
        print(adjusted_threshold, "Y:", average_y, "X:", average_x, "Z:", average_z)
        return adjusted_threshold