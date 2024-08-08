import cv2
from forAllTest import init
import mediapipe as mp

init()
from FrameExtractorClass import FrameExtractor
from pointsDetectorClass import pointsDetector

# image_path = 'C:\\Users\\48113164\\Documents\\GitHub\\SignAI-IA.dev\\AI-ML_Development\\recursos\\4.jpg'
# frame = cv2.imread(image_path)
# FE = FrameExtractor()
# FE.adjust_threshold(frame)

video = cv2.VideoCapture('C:\\Users\\48113164\\Documents\\GitHub\\SignAI-IA.dev\\AI-ML_Development\\recursos\\videoprueba4.mp4')

FE = FrameExtractor()
Frames = FE.extractKeyFrames(1,1,video)
print(len(Frames))
for frame in Frames:
    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()