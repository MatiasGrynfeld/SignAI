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
        vectors = []
        for hand in landmarks:
            hand_points = hand[0].landmark
            hand_points_vector = np.array([[point.x, point.y, point.z] for point in hand_points])
            hand_points_vector[:, 0] = self.normalize_vector(hand_points_vector[:, 0])
            hand_points_vector[:, 1] = self.normalize_vector(hand_points_vector[:, 1])
            hand_points_vector[:, 2] = self.normalize_vector(hand_points_vector[:, 2])
            vectors.append(hand_points_vector)
        return vectors
