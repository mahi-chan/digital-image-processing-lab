import cv2
import numpy as np
import matplotlib.pyplot as plt

def log_transform(image):
    c = 255 / (np.log(1 + np.max(image)))
    log_image = c * np.log(1 + image)
    log_image = np.array(log_image, dtype=np.uint8)
    return log_image

def gamma_transform(image, gamma):
    gamma_corrected = np.array(255 * (image / 255) ** gamma, dtype='uint8')
    return gamma_corrected

image_path = 'R1.jpg'
image = cv2.imread(image_path)

if len(image.shape) == 2:
    log_image = log_transform(image)
    gamma_image = gamma_transform(image, 2.2)
else:  # Color image
    b, g, r = cv2.split(image)
    log_b = log_transform(b)
    log_g = log_transform(g)
    log_r = log_transform(r)
    log_image = cv2.merge((log_b, log_g, log_r))
    gamma_b = gamma_transform(b, 2.2)
    gamma_g = gamma_transform(g, 2.2)
    gamma_r = gamma_transform(r, 2.2)
    gamma_image = cv2.merge((gamma_b, gamma_g, gamma_r))

plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.title('Original Image')
if len(image.shape) == 2:
    plt.imshow(image, cmap='gray')
else:
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(1, 3, 2)
plt.title('Log Transformed Image')
if len(log_image.shape) == 2:
    plt.imshow(log_image, cmap='gray')
else:
    plt.imshow(cv2.cvtColor(log_image, cv2.COLOR_BGR2RGB))

plt.subplot(1, 3, 3)
plt.title('Gamma Transformed Image')
if len(gamma_image.shape) == 2:
    plt.imshow(gamma_image, cmap='gray')
else:
    plt.imshow(cv2.cvtColor(gamma_image, cv2.COLOR_BGR2RGB))

plt.show()
