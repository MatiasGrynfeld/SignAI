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
        vector.append(self.hand2vec(landmarks[0]))
        vector.append(self.hand2vec(landmarks[1]))
        vector.append(self.face2vec(landmarks[2]))
        vector.append(self.pose2vec(landmarks[3]))
        return vector
    
    def hand2vec(self, hand):
        hand_points = hand.landmark
        hand_points_vector = np.array([[point.x, point.y, point.z] for point in hand_points])
        hand_points_vector[:, 0] = self.normalize_vector(hand_points_vector[:, 0])
        hand_points_vector[:, 1] = self.normalize_vector(hand_points_vector[:, 1])
        hand_points_vector[:, 2] = self.normalize_vector(hand_points_vector[:, 2])
        return hand_points_vector
    
    def face2vec(self, face):
        
        face_points = face.landmark
        face_points_vector = np.array([[point.x, point.y, point.z] for point in face_points])
        face_points_vector[:, 0] = self.normalize_vector(face_points_vector[:, 0])
        face_points_vector[:, 1] = self.normalize_vector(face_points_vector[:, 1])
        face_points_vector[:, 2] = self.normalize_vector(face_points_vector[:, 2])
        return face_points_vector
    
    def pose2vec(self, pose):
        pose_points = pose.landmark
        pose_points_vector = np.array([[point.x, point.y, point.z] for point in pose_points])
        pose_points_vector[:, 0] = self.normalize_vector(pose_points_vector[:, 0])
        pose_points_vector[:, 1] = self.normalize_vector(pose_points_vector[:, 1])
        pose_points_vector[:, 2] = self.normalize_vector(pose_points_vector[:, 2])
        return pose_points_vector
