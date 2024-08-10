from forAllTest import init
import cv2
init()
from PoseDetectorClass import PoseDetector

poseDetector = PoseDetector(is_image=True, num_hands=2, detection_confidence=0.5, tracking_confidence=0.5)
fotoPath = 'a.jpg'
foto = cv2.imread(fotoPath)
results = poseDetector.extractPoints(foto)
print(results[0].multi_hand_landmarks, results[1].pose_landmarks)
foto = poseDetector.drawLandmarks(results, foto)
cv2.imshow("Image", foto)
cv2.waitKey(0)
cv2.destroyAllWindows()