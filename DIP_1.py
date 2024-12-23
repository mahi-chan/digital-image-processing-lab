from PIL import Image
import numpy as np

def load_image_as_array(image_path, mode='RGB'):
    with Image.open(image_path) as img:
        if mode:
            img = img.convert(mode)
        return np.array(img)


gray_image_path = 'R1.jpg'
color_image_path = 'R.jpeg'

gray_image_array = load_image_as_array(gray_image_path, mode='L')
color_image_array = load_image_as_array(color_image_path, mode='RGB')

print("Grayscale Image Array (2D):")
print(gray_image_array)

print("\nColor Image Array (3D):")
print(color_image_array)
