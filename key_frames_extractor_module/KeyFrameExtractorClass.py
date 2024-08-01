from ..hand_detection_module.HandDetectorClass import HandDetector

class KeyFrameExtractor:
    def __init__(self) -> None:
        self.hand_detector = HandDetector(is_image=True, num_hands=2, detection_confidence=0.5, tracking_confidence=0.5)
    
    def calcularDiferencia(self, puntos_previos, puntos_actuales, handedness_prev, handedness_actual):
        if puntos_previos is None or puntos_actuales is None:
            return 10.0

        # Comprobamos si alguna de las listas está vacía o si las dos listas tienen diferentes longitudes
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

    def extractKeyFrames(self, video, min_frame_interval=3):
        puntos_previos = []
        handedness_prev = []
        key_frames = []
        frame_count = 0

        for frame in self.extractFrames(video):
            frame_count += 1
            if len(key_frames)==0:
                if results.multi_hand_landmarks:
                    handedness_prev = results.multi_handedness if results.multi_handedness else []
                    puntos_previos=results.multi_hand_landmarks
                    threshold = self.adjust_threshold(puntos_previos)
                    key_frames.append((frame, frame_count))
                    print("Threshold:", threshold)

            elif frame_count - key_frames[-1][1] > min_frame_interval:
                results = self.hand_detector.detectHands(frame)
                if results.multi_hand_landmarks:
                    puntos_actuales = results.multi_hand_landmarks
                    handedness_actual = results.multi_handedness if results.multi_handedness else []
                    diff = self.calcularDiferencia(puntos_previos, puntos_actuales, handedness_prev, handedness_actual)

                    if diff > threshold:
                        key_frames.append((frame, frame_count))
                        puntos_previos = puntos_actuales
                        handedness_prev = handedness_actual
                else:
                    puntos_previos = None
        return [frame for frame, _ in key_frames]
    
    def adjust_threshold(self, puntos_actuales, base_threshold=40.0):
        for hand_landmarks in puntos_actuales:
            x_coords = [landmark.x for landmark in hand_landmarks.landmark]
            y_coords = [landmark.y for landmark in hand_landmarks.landmark]
        hand_width = max(x_coords) - min(x_coords)
        hand_height = max(y_coords) - min(y_coords)
        hands_size= hand_width * hand_height
        threshold=base_threshold*hands_size
        print (threshold)
        return threshold