import numpy as np

class Point2Vec:
    def __init__(self, numDecimals = 4):
        self.numDecimals = numDecimals
    
    def truncate(self, number):
        comaIndex = 10 ** self.numDecimals
        return np.round(number * comaIndex) / comaIndex
    
    def normalize(self, value, min, max):
        """Min-Max normalization"""
        value = self.truncate(value)
        return (value-min)/(max-min)
    
    def normalizeVector(self, vector):
        min = np.min(vector)
        max = np.max(vector)
        normalizedVector = []
        for number in vector:
            normalizedVector.append(self.normalize(number, min, max))
        
        return normalizedVector
    
    def land2Vec(self, landmarks):
        vector = []
        for hand in landmarks:
            handPoints = hand[0].landmark
            handPointsVector = [[point.x, point.y, point.z] for point in handPoints]
            handPointsVector[:,0] = self.normalizeVector(handPointsVector[:,0])
            handPointsVector[:,1] = self.normalizeVector(handPointsVector[:,1])
            handPointsVector[:,2] = self.normalizeVector(handPointsVector[:,2])
            vector.append(handPointsVector)
        return vector