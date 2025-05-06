from matplotlib import pyplot as plt
from scipy.signal import argrelextrema

import motion_detection_by_pixels_changes
import corner_detection
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

num_of_pages = 8
motion_detection_by_pixels_changes.video_to_images(r"C:\Users\user\Desktop\project\0.mp4", num_of_pages)




file_names = os.listdir(r"C:\Users\user\PycharmProjects\pythonProject10\selected_images")

# עיבוד כל קובץ בתיקייה


def filter_minima_by_distance(minima_indices, projection, min_distance=100):
    """Filters the minima to ensure they are at least min_distance apart."""
    if len(minima_indices) < 3:
        return minima_indices  # If there aren't enough points, return as is.

    # Sort minima by their projection value (lowest first)
    sorted_minima = sorted(minima_indices, key=lambda x: projection[x])

    # Select only those that are far enough apart
    selected_minima = [sorted_minima[0]]  # Start with the lowest point

    for min_index in sorted_minima[1:]:
        if all(abs(min_index - sel) > min_distance for sel in selected_minima):
            selected_minima.append(min_index)
        if len(selected_minima) == 3:
            break  # Stop when we have 3 valid minima

    # Sort the selected minima **by index order in the image**
    selected_minima.sort()

    return selected_minima

def page_split(image_path, num=""):
    # Load image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_colored = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.GaussianBlur(image, (5, 5), 0.5)
    if image is None:
        raise ValueError("Image not found! Check the file path.")

    # Step 1: Thresholding
    _, binary = cv2.threshold(image, 148, 255, cv2.THRESH_BINARY_INV)

    # Step 2: Compute brightness histogram along the vertical axis
    projection = np.sum(binary, axis=0)
    length = len(projection)
    maxx = 0
    right = 0
    left = 0
    for ind,i in enumerate(projection[:length//2]):
        if i > maxx:
            maxx = i
        if i == maxx or i > maxx * 0.98:
            right = ind
    print(right)
    for ind,i in enumerate(projection[length -1 :length//2:-1]):
        if i > maxx:
            maxx = i
        if i == maxx or i > maxx * 0.96:
            left = len(projection) - ind
    # Find all local minima
    # minima_indices = argrelextrema(projection, np.less)[0]
    # valid_minima = filter_minima_by_distance(minima_indices, projection, min_distance=300)
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 3, 1)
    # plt.title("Original Image")
    # plt.imshow(image, cmap="gray")
    #
    # plt.subplot(1, 3, 2)
    # plt.title("Binary Image")
    # plt.imshow(binary, cmap="gray")
    #
    # plt.subplot(1, 3, 3)
    # plt.title("Projection Profile")
    # plt.plot(projection)
    middle = (left + right) // 2
    # print(middle)
    # for min_idx in valid_minima:
    #     plt.axvline(min_idx, color='g', linestyle='--', label=f"Min {min_idx}")
    # plt.axvline(middle, color='r', linestyle='--', label=f'Selected Min: {middle}')
    # plt.legend()
    # plt.show()

    # Split the image, adding 20 pixels from the other side
    left_page = image_colored[:, :middle + 30]
    right_page = image_colored[:, middle - 30:]

    if not os.path.exists("selected_imagesA"):
        os.makedirs("selected_imagesA")

    # Save both pages
    file_path_left = os.path.join("selected_imagesA", f"left_page_{num}")
    file_path_right = os.path.join("selected_imagesA", f"right_page_{num}")
    cv2.imwrite(file_path_left , left_page)
    cv2.imwrite(file_path_right, right_page)


    print(f"Selected middle minimum by index found at column: {middle}")



for file_name in file_names:
    file_path = os.path.join(r"C:\Users\user\PycharmProjects\pythonProject10\selected_images", file_name)
    page_split(file_path,file_name)


count = 0
file_names = os.listdir(r"C:\Users\user\PycharmProjects\pythonProject10\selected_imagesA")
for file_name in file_names:
    count += 1
    file_path = os.path.join(r"C:\Users\user\PycharmProjects\pythonProject10\selected_imagesA", file_name)
    if count < 18:
        corner_detection.analyze_page(file_path,1,file_name)
    else:
        corner_detection.analyze_page(file_path, 0,file_name)
