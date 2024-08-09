import numpy as np

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
        return self.truncate(normalized_vector)
    
    def land2vec(self, landmarks):
        """Convert MediaPipe landmarks to normalized vector representation"""
        vector = []
        for keyFrame in landmarks:
            subVector = [
                self.hand2vec(keyFrame[0]),
                self.hand2vec(keyFrame[1]),
                self.face2vec(keyFrame[2]),
                self.pose2vec(keyFrame[3])
            ]
            vector.append(subVector)
        return vector
    
    def hand2vec(self, hand):
        if hand:
            hand_points = hand.landmark
            hand_points_vector = np.array([[point.x, point.y, point.z] for point in hand_points])
            hand_points_vector[:, 0] = self.normalize_vector(hand_points_vector[:, 0])
            hand_points_vector[:, 1] = self.normalize_vector(hand_points_vector[:, 1])
            hand_points_vector[:, 2] = self.normalize_vector(hand_points_vector[:, 2])
        else:
            hand_points_vector = [[-1, -1, -1] * 21]
        return hand_points_vector
    
    def face2vec(self, face):
        if face:
            face_points = face.landmark
            face_points_vector = np.array([[point.x, point.y, point.z] for point in face_points])
            face_points_vector[:, 0] = self.normalize_vector(face_points_vector[:, 0])
            face_points_vector[:, 1] = self.normalize_vector(face_points_vector[:, 1])
            face_points_vector[:, 2] = self.normalize_vector(face_points_vector[:, 2])
        else:
            face_points_vector = [[-1, -1 , -1] * 468]
        return face_points_vector
    
    def pose2vec(self, pose):
        if pose:
            pose_points = pose.landmark
            pose_points_vector = np.array([[point.x, point.y, point.z] for point in pose_points])
            pose_points_vector[:, 0] = self.normalize_vector(pose_points_vector[:, 0])
            pose_points_vector[:, 1] = self.normalize_vector(pose_points_vector[:, 1])
            pose_points_vector[:, 2] = self.normalize_vector(pose_points_vector[:, 2])
        else:
            pose_points_vector = [[-1, -1, -1] * 33] 
        return pose_points_vector
