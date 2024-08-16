import numpy as np
np.set_printoptions(suppress=True, precision=4)
class Point2Vec:
    def __init__(self, num_decimals=4):
        self.num_decimals = num_decimals
    
    def truncate(self, number):
        factor = 10 ** self.num_decimals
        return np.round(number * factor) / factor
    
    def normalize(self, value, min_val, max_val):
        """Min-Max normalization"""
        return self.truncate((value - min_val) / (max_val - min_val))
    
    def normalize_vector(self, vector):
        min_val = np.min(vector)
        max_val = np.max(vector)
        normalized_vector = (vector - min_val) / (max_val - min_val)
        factor = 10 ** self.num_decimals
        truncated_vector = np.round(normalized_vector * factor) / factor
        return truncated_vector


    def land2vec(self, landmarks):
        """Convert MediaPipe landmarks to normalized vector representation"""
        vectors = []
        for keyFrame in landmarks:
            handLandmarks = keyFrame[0]
            poseLandmarks = keyFrame[1]
            vectors.append(
                np.concatenate([
                self.hands2vec(handLandmarks),
                self.pose2vec(poseLandmarks)
                ])
            )
        return vectors

    def hands2vec(self, hand):
        if hand:
            if len(hand.multi_hand_landmarks) == 1:
                hand_points_vector = np.concatenate([
                    self.hand2vec(hand.multi_hand_landmarks[0]),
                    self.hand2vec(None)
                ])
            elif len(hand.multi_hand_landmarks) == 2:
                hand_points_vector = np.concatenate([
                    self.hand2vec(hand.multi_hand_landmarks[0]),
                    self.hand2vec(hand.multi_hand_landmarks[1])
                ])
            else:
                hand_points_vector = np.concatenate([
                    self.hand2vec(None),
                    self.hand2vec(None)
                ])
        else:
            hand_points_vector = np.concatenate([
                self.hand2vec(None),
                self.hand2vec(None)
            ])
        return hand_points_vector

    
    def hand2vec(self, hand):
        if hand:
            hand_points_vector = np.array([[point.x, point.y, point.z, 1] for point in hand.landmark])
            hand_points_vector[:, 0] = self.normalize_vector(hand_points_vector[:, 0])
            hand_points_vector[:, 1] = self.normalize_vector(hand_points_vector[:, 1])
            hand_points_vector[:, 2] = self.normalize_vector(hand_points_vector[:, 2])
        else:
            hand_points_vector = np.full((21, 4), [-1, -1, -1, 0])
        return hand_points_vector.flatten()

    def pose2vec(self, pose):
        if pose:
            pose_points_vector = np.array([[point.x, point.y, point.z, point.visibility] for point in pose.pose_landmarks.landmark])
            pose_points_vector[:, 0] = self.normalize_vector(pose_points_vector[:, 0])
            pose_points_vector[:, 1] = self.normalize_vector(pose_points_vector[:, 1])
            pose_points_vector[:, 2] = self.normalize_vector(pose_points_vector[:, 2])
        else:
            pose_points_vector = np.full((33, 4), [-1, -1, -1, 0])
        return pose_points_vector.flatten()
    
    def CNNMaxtrix(self, landmarks):
        vector = self.land2vec(landmarks)
        returnVector = []
        for keyFrame in vector:
            keyFrame = keyFrame.reshape(15,20)
            zeros_5x20 = np.zeros((5, 20))
            returnVector.append(np.vstack((keyFrame, zeros_5x20)))
        return returnVector
