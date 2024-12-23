import cv2
import numpy as np

image = cv2.imread('R1.jpg', cv2.IMREAD_GRAYSCALE)

kernel = np.array([[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]], dtype=np.float32) / 9

rows, cols = image.shape
output = np.zeros_like(image, dtype=np.uint8)

for i in range(1, rows - 1):
    for j in range(1, cols - 1):
        neighborhood = image[i - 1:i + 2, j - 1:j + 2]
        # print(neighborhood)
        result = np.sum(neighborhood * kernel)
        output[i, j] = result.astype(np.uint8)

# cv2.imwrite('blurred_image.jpg', output)
cv2.imshow('Original Image', image)
cv2.imshow('Blurred Image', output)
cv2.waitKey(0)
cv2.destroyAllWindows()