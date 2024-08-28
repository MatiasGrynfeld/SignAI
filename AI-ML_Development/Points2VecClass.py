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
            handLandmarks = [keyFrame.left_hand_landmarks, keyFrame.right_hand_landmarks]
            poseLandmarks = keyFrame.pose_landmarks
            faceLandmarks = keyFrame.face_landmarks
            vectors.append(
                np.concatenate([
                self.hands2vec(handLandmarks),
                self.pose2vec(poseLandmarks),
                self.face2vec(faceLandmarks)
                ]).tolist()
            )
        return vectors

    def hands2vec(self, hand):
        if hand:
            if hand[0] and not hand[1]:
                hand_points_vector = np.concatenate([
                    self.hand2vec(hand[0]),
                    self.hand2vec(None)
                ])
            elif not hand[0] and hand[1]:
                hand_points_vector = np.concatenate([
                    self.hand2vec(None),
                    self.hand2vec(hand[1])
                ])
            elif hand[0] and hand[1]:
                hand_points_vector = np.concatenate([
                    self.hand2vec(hand[0]),
                    self.hand2vec(hand[1])
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
            pose_points_vector = np.array([[point.x, point.y, point.z, point.visibility] for point in pose.landmark])
            pose_points_vector[:, 0] = self.normalize_vector(pose_points_vector[:, 0])
            pose_points_vector[:, 1] = self.normalize_vector(pose_points_vector[:, 1])
            pose_points_vector[:, 2] = self.normalize_vector(pose_points_vector[:, 2])
        else:
            pose_points_vector = np.full((33, 4), [-1, -1, -1, 0])
        return pose_points_vector.flatten()
    
    def face2vec(self, face):
        if face:
            face_points_vector = np.array([[point.x, point.y, point.z, 1] for point in face.landmark])
            face_points_vector[:, 0] = self.normalize_vector(face_points_vector[:, 0])
            face_points_vector[:, 1] = self.normalize_vector(face_points_vector[:, 1])
            face_points_vector[:, 2] = self.normalize_vector(face_points_vector[:, 2])
        else:
            face_points_vector = np.full((468, 4), [-1, -1, -1, 0])
        return face_points_vector.flatten()
    
    def CNNMatrix(self, landmarks):
        vector = self.land2vec(landmarks)
        returnVector = []
        for keyFrame in vector:
            zero_row = [0] * 36
            keyFrame = np.concatenate([keyFrame, zero_row])
            keyFrame = keyFrame.reshape(46,48)
            zeros_5x20 = np.zeros((2, 48))
            returnVector.append(np.vstack((keyFrame, zeros_5x20)))
        return returnVector
