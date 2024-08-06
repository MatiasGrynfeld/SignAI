import numpy as np

class VectorNormalizer:
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