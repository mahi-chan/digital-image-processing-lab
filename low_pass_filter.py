import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('R1.jpg', cv2.IMREAD_GRAYSCALE)

# Get the dimensions of the image
rows, cols = image.shape

# Create a meshgrid
x, y = np.meshgrid(np.arange(cols), np.arange(rows))

# Calculate the distance from the center of the image
center_x, center_y = cols // 2, rows // 2
distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

# Define the Butterworth filter parameters
threshold_distance = 300
order = 3

# Create the Butterworth filter
butterworth_filter = 1 / (1 + (distance / threshold_distance) ** (2 * order))

# Define the Laplacian kernel
laplacian_kernel = np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]])

# Apply the Laplacian kernel to the image
laplacian_image = cv2.filter2D(image, -1, laplacian_kernel)

# Apply the Butterworth filter to the Laplacian image
filtered_image = laplacian_image * butterworth_filter

# Normalize the filtered image
filtered_image = np.clip(filtered_image, 0, 255)

# Convert to uint8
filtered_image = filtered_image.astype(np.uint8)

# Display the original and filtered images
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(122)
plt.imshow(filtered_image, cmap='gray')
plt.title(f'Filtered Image (Threshold Distance: {threshold_distance})')

plt.show()
