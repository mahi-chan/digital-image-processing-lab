import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_histogram(img):
    hist = np.zeros(256, dtype=int)
    for px in img.flatten():
        hist[px] += 1
    return hist

def calculate_cumulative_histogram(hist):
    return np.cumsum(hist)

def equalize_histogram(img):
    hist = calculate_histogram(img)
    cum_hist = calculate_cumulative_histogram(hist)
    eq_img = np.zeros_like(img)
    total_pixels = img.size
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            eq_img[row, col] = (cum_hist[img[row, col]] * 255) // total_pixels
    return eq_img

img = cv2.imread('R.jpeg', cv2.IMREAD_GRAYSCALE)

orig_hist = calculate_histogram(img)
eq_img = equalize_histogram(img)
eq_hist = calculate_histogram(eq_img)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.bar(range(256), orig_hist)
plt.title("Original Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
plt.bar(range(256), eq_hist)
plt.title("Equalized Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")

plt.tight_layout()

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(eq_img, cmap='gray')
plt.title("Equalized Image")

plt.tight_layout()
plt.show()
