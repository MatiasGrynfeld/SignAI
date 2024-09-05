import cv2
from forAllTest import init
import os

init()
from KeyFrameExtractorClass import KeyFrameExtractor
from Points2VecClass import Point2Vec

def obtain_paths(root, extension):
    paths = []

    for filename in os.listdir(root):
        if filename.endswith(extension):
            paths.append(os.path.join(root, filename))
    return paths

vids = obtain_paths("C:\\Users\\48519558\\Desktop\\SignAI-ML\\AI-ML_Development\\Resources", '.mp4')
print(vids)

kfe = KeyFrameExtractor()
p2v = Point2Vec()

for video in vids:
    vid = cv2.VideoCapture(video)
    frames = kfe.extractKeyFrames(return_frame=True, draw=True, video=vid)
    print(len(frames))
    for frame in frames:
        cv2.imshow('frame', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()