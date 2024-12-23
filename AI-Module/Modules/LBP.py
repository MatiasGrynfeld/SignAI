import numpy as np
from skimage import feature

def LBP(image: np.ndarray, numPoints: int, radius: int, crop_width: int) -> np.ndarray:
    if crop_width != 0:
        h, w = image.shape
        start_w = max((w - crop_width) // 2, 0)
        end_w = min(start_w + crop_width, w)
        cropped_image = image[:, start_w:end_w]
        lbp = feature.local_binary_pattern(cropped_image, numPoints, radius, method="uniform")
    else:
        lbp = feature.local_binary_pattern(image, numPoints, radius, method="uniform")
    return lbp

def LBP_manual(image: np.ndarray):
    pass

if __name__ == '__main__':
    from pathlib import Path
    import cv2
    import matplotlib.pyplot as plt
    import os

    input_image = cv2.imread(
        str(Path(os.getcwd()) / 'AI-Module' / 'Resources' / 'Videos' / 'Frames' / 'start' / 'frame_0.jpg'), 
        cv2.IMREAD_GRAYSCALE
    )

    lbp_image = LBP(input_image, numPoints=8, radius=1, crop_width=500)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Imagen original")
    plt.imshow(input_image, cmap='gray')
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Imagen LBP")
    plt.imshow(lbp_image, cmap='gray')
    plt.axis("off")

    plt.tight_layout()
    plt.show()
