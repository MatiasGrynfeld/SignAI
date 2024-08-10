# from PoseDetectorClass import PoseDetector
# class KeyFrameExtractor:
#     def __init__(self) -> None:
#         self.poseDetector = PoseDetector(is_image=True, num_hands=2, detection_confidence=0.5, tracking_confidence=0.5)
    
#     def calcularDiferencia(self, puntos_previos, puntos_actuales, hands_detected_prev, hands_detected_actual):
#         if puntos_previos is None:
#             return 100.0
#         if puntos_actuales is None:
#             return 0.0
#         if hands_detected_prev!=hands_detected_actual:
#             return 100.0
#         right_hand_prev = puntos_previos[:21] if hands_detected_prev >= 1 else None
#         left_hand_prev = puntos_previos[21:42] if hands_detected_prev == 2 else None
#         right_hand_actual = puntos_actuales[:21] if hands_detected_actual >= 1 else None
#         left_hand_actual = puntos_actuales[21:42] if hands_detected_actual == 2 else None
#         diff=self.calcularDifManos(right_hand_prev,right_hand_actual,left_hand_prev,left_hand_actual,hands_detected_actual)
#         return float(diff)
    
#     def calcularDifManos(self, right_hand_prev, right_hand_actual, left_hand_prev, left_hand_actual, num_hands):
#         total_diff = 0.0
        
#         def calc_diff(hand_prev, hand_actual):
#             return sum([
#                 abs(lm1.x - lm2.x) + abs(lm1.y - lm2.y) + abs(lm1.z - lm2.z)
#                 for lm1, lm2 in zip(hand_prev, hand_actual)
#             ])
        
#         if num_hands == 2:
#             if right_hand_prev and right_hand_actual:
#                 total_diff += calc_diff(right_hand_prev, right_hand_actual)
#             if left_hand_prev and left_hand_actual:
#                 total_diff += calc_diff(left_hand_prev, left_hand_actual)
#             total_diff /= 2
            
#         elif num_hands == 1:
#             if left_hand_prev and left_hand_actual:
#                 total_diff = calc_diff(left_hand_prev, left_hand_actual)
#             elif right_hand_prev and right_hand_actual:
#                 total_diff = calc_diff(right_hand_prev, right_hand_actual)
#             else:
#                 return 100.0
        
#         return total_diff

#     def extractKeyFrames(self, return_frame, draw, video, min_frame_interval=1):
#         puntos_previos = []
#         manos_detectadas_actual=0
#         key_frames = []
#         frame_count = 0
#         ret = 1
#         while ret:
#             ret, frame = video.read()
#             if not ret:
#                 break
#             frame_count += 1
#             if len(key_frames)==0:
#                 results = self.poseDetector.extractPoints(frame)
#                 if results[0].multi_hand_landmarks:
#                     puntos_previos = results[0].multi_hand_landmarks
#                     manos_detectadas_prev = len(results[0].multi_hand_landmarks) // 21
#                     threshold = self.adjust_threshold(puntos_previos)
#                     if return_frame:
#                         if draw:
#                             frame = self.poseDetector.drawLandmarks(puntos_previos, frame)
#                         key_frames.append((frame, frame_count))
#                     else:
#                         key_frames.append((results, frame_count))

#             elif frame_count - key_frames[-1][1] > min_frame_interval:
#                 results = self.poseDetector.extractPoints(frame)
#                 if results[0].multi_hand_landmarks:
#                     puntos_actuales = results[0].multi_hand_landmarks
#                     manos_detectadas_actual = len(results[0].multi_hand_landmarks) // 21
#                     diff = self.calcularDiferencia(puntos_previos, puntos_actuales, manos_detectadas_prev, manos_detectadas_actual)

#                     if diff > threshold:
#                         if return_frame:
#                             if draw:
#                                 frame = self.poseDetector.drawLandmarks(puntos_actuales, frame)
#                             key_frames.append((frame, frame_count))
#                         else:
#                             key_frames.append((results, frame_count))
#                             puntos_previos = puntos_actuales
#                             manos_detectadas_prev = manos_detectadas_actual
#                 else:
#                     puntos_previos = None
#         video.release()
#         return [frame for frame, _ in key_frames]

#     def adjust_threshold(self, puntos, base_threshold=2.0):
#         y_values = []
#         z_values=[]
#         x_values=[]
#         ysum=0
#         xsum=0
#         zsum=0
#         if puntos[:21]:
#             for z in puntos[:21]:
#                 ysum += z.y
#                 xsum += z.x
#                 zsum += z.z
#             y_values.append(ysum / len(puntos[:21]))
#             x_values.append(xsum / len(puntos[:21]))
#             z_values.append(zsum / len(puntos[:21]))
            
#         if puntos[21:42]:
#             for z in puntos[21:42]:
#                 ysum += z.y
#                 xsum += z.x
#                 zsum += z.z
#             y_values.append(ysum / len(puntos[21:42]))
#             x_values.append(xsum / len(puntos[21:42]))
#             z_values.append(zsum / len(puntos[21:42]))
        
#         if not z_values:
#             return base_threshold
        
#         average_y = sum(y_values) / len(y_values)
#         average_x = sum(x_values) / len(x_values)
#         average_z = sum(z_values) / len(z_values)
#         adjusted_threshold = base_threshold + abs(average_z * 50 * base_threshold)
#         print(adjusted_threshold)
#         return adjusted_threshold

from PoseDetectorClass import PoseDetector
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
        return [frame for frame, _ in key_frames]
    
    def adjust_threshold(self, puntos_actuales, base_threshold=40.0):
        for hand_landmarks in puntos_actuales:
            x_coords = [landmark.x for landmark in hand_landmarks.landmark]
            y_coords = [landmark.y for landmark in hand_landmarks.landmark]
        hand_width = max(x_coords) - min(x_coords)
        hand_height = max(y_coords) - min(y_coords)
        hands_size= hand_width * hand_height
        threshold=base_threshold*hands_size
        return threshold