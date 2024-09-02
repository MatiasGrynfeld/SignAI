import cv2
from forAllTest import init

init()
from KeyFrameExtractorClass import KeyFrameExtractor
from Points2VecClass import Point2Vec

#video = cv2.VideoCapture('C:\\Users\\48519558\\Desktop\\SignAI-ML\\AI-ML_Development\\Resources\\Videos\\5RR7dlzkyL0-5-rgb_front.mp4')
video = cv2.VideoCapture('C:\\Users\\48519558\\Desktop\\SignAI-ML\\AI-ML_Development\\Resources\\_2FBDaOPYig-5-rgb_front.mp4')
KFE = KeyFrameExtractor()

keyFrames = KFE.extractKeyFrames(return_frame=True, draw=True, video=video)
# print("segundo video")
# keyFrames2 = KFE.extractKeyFrames(return_frame=True, draw=True, video=video2)
# p2v = Point2Vec(4)
# keyFrames = p2v.land2vec(keyFrames)
# print(len(keyFrames))
# print(keyFrames)
for frame in keyFrames:
    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# print(len(keyFrames2))
# for frame in keyFrames2:
#     cv2.imshow('frame', frame)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()