import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_histogram(img):
    hist = np.zeros(256, dtype=np.uint8)
    for pixel in img.flatten():
        hist[pixel] += 1
    return hist

def calculate_cumulative_histogram(hist):
    return np.cumsum(hist)

def match_histograms(src_img, ref_img):
    src_hist = calculate_histogram(src_img)
    ref_hist = calculate_histogram(ref_img)

    src_cdf = calculate_cumulative_histogram(src_hist)
    ref_cdf = calculate_cumulative_histogram(ref_hist)

    lookup_table = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        lookup_table[i] = np.argmin(np.abs(ref_cdf - src_cdf[i]))

    matched_img = np.zeros_like(src_img)
    for row in range(src_img.shape[0]):
        for col in range(src_img.shape[1]):
            matched_img[row, col] = lookup_table[src_img[row, col]]

    return matched_img

def display_images_and_histograms(src_hist, ref_hist, matched_hist, src_img, ref_img, matched_img):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.bar(range(256), src_hist)
    plt.title("Source Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")

    plt.subplot(1, 3, 2)
    plt.bar(range(256), ref_hist)
    plt.title("Reference Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")

    plt.subplot(1, 3, 3)
    plt.bar(range(256), matched_hist)
    plt.title("Matched Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(src_img, cmap='gray')
    plt.title("Source Image")

    plt.subplot(1, 3, 2)
    plt.imshow(ref_img, cmap='gray')
    plt.title("Reference Image")

    plt.subplot(1, 3, 3)
    plt.imshow(matched_img, cmap='gray')
    plt.title("Matched Image")

    plt.tight_layout()
    plt.show()

src_img = cv2.imread('R.jpeg', 0)
ref_img = cv2.imread('R1.jpg', 0)

matched_img = match_histograms(src_img, ref_img)

src_hist = calculate_histogram(src_img)
ref_hist = calculate_histogram(ref_img)
matched_hist = calculate_histogram(matched_img)

display_images_and_histograms(src_hist, ref_hist, matched_hist, src_img, ref_img, matched_img)
