import cv2
from forAllTest import init

init()
from FrameExtractorClass import FrameExtractor
from pointsDetectorClass import pointsDetector

video = cv2.VideoCapture('C:\\Users\\48113164\\Documents\\GitHub\\SignAI-IA.dev\\AI-ML_Development\\recursos')
FE = FrameExtractor()
Frames = FE.extractFrames(video)
print(len(Frames))
for frame in Frames:
    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()