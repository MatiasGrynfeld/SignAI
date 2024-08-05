import cv2
from forAllTest import init

init()
from KeyFrameExtractorClass import KeyFrameExtractor

video = cv2.VideoCapture('C:\\Users\\matia\\OneDrive\\Escritorio\\SignAI-ML\\AI-ML_Development\\recursos\\videoprueba4.mp4')
KFE = KeyFrameExtractor()

keyFrames = KFE.extractKeyFrames(video)
print(len(keyFrames))
for frame in keyFrames:
    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()