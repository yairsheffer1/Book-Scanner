import cv2
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from helpers.utils import show_img

# ---------- פונקציות עזר ----------
def draw_corners(corners, image, is_left_page=True):
    color = (0, 255, 0)
    labels = ["Left Top", "Left Bottom", "Right Bottom", "Right Top"]

    for (x, y), label in zip(corners, labels):
        if x != -1 and y != -1:
            print(f"{label} Corner: ({x}, {y})")
            cv2.circle(image, (x, y), 10, color, -1)
        else:
            print(f"{label} Corner: Not found")

    plt.figure(figsize=(12, 6))
    title = "Left Page" if is_left_page else "Right Page"
    plt.title(f"{title} - Detected Corners")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


def find_y(binary, threshold_percent, from_top=True):
    height, width = binary.shape
    y_range = range(height) if from_top else range(height - 1, -1, -1)
    for y in y_range:
        row = binary[y, :]
        black_pixel_ratio = np.sum(row) / (255 * width) * 100
        if black_pixel_ratio <= threshold_percent:
            return y
    return -1


def find_x(binary,threshold_percent,dir="a"):
    height, width = binary.shape

    if dir == "left_to_right":
        x_range = range(width)  # 0 עד width-1
    else:
        x_range = range(width - 1, -1, -1)

    for x in x_range:
        col = binary[:, x]
        black_pixel_ratio = np.sum(col) / (255 * height) * 100
        if black_pixel_ratio <= threshold_percent:
            return x
    return -1


def crop_image(corners,image_colored):
    if all(c != -1 for pt in corners for c in pt):
        h, w = image_colored.shape[:2]
        xs = [pt[0] for pt in corners]
        ys = [pt[1] for pt in corners]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # הרחבות מוצעות
        padding_x = 30
        padding_y = 20

        # הרחבת ציר X שמאלה
        if min_x - padding_x >= 0:
            min_x -= padding_x

        # הרחבת ציר X ימינה
        if max_x + padding_x <= w:
            max_x += padding_x

        # הרחבת ציר Y למעלה
        if min_y - padding_y >= 0:
            min_y -= padding_y

        # הרחבת ציר Y למטה
        if max_y + padding_y <= h:
            max_y += padding_y

        # בדיקה שהגבולות עדיין תקינים
        if max_x > min_x and max_y > min_y:
            cropped = image_colored[min_y:max_y, min_x:max_x]

    cropped = image_colored[min_y:max_y, min_x:max_x]
    return cropped

# ---------- ניתוח עמוד ----------

def analyze_page(image_path, is_left_page=False,num=""):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_colored = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Image not found! Check the file path.")

    image = cv2.GaussianBlur(image, (5, 5), 0.5)
    _, binary = cv2.threshold(image, 195, 255, cv2.THRESH_BINARY_INV)

    if is_left_page:
        # -------- עמוד שמאל --------
        right_bottom_x = binary.shape[1] - 1
        right_bottom_y = find_y(binary, 90, from_top=False)

        right_top_x = binary.shape[1] - 1
        right_top_y = find_y(binary, 90, from_top=True)

        left_top_x = find_x(binary, 90,"left_to_right")
        left_top_y = find_y(binary, 90, from_top=True)

        left_bottom_x = find_x(binary, 90,"left_to_right")
        left_bottom_y = find_y(binary, 90, from_top=False)

    else:
        # -------- עמוד ימין --------
        right_bottom_x = find_x(binary, 90)
        right_bottom_y = find_y(binary, 90, from_top=False)

        right_top_x = find_x(binary, 90)
        right_top_y = find_y(binary, 90, from_top=True)

        left_top_x = 0
        left_top_y = find_y(binary, 90, from_top=True)

        left_bottom_x = 0
        left_bottom_y = find_y(binary, 90, from_top=False)

    if is_left_page:
        corners = [
            (left_top_x, left_top_y),
            (left_bottom_x, left_bottom_y),
            (right_bottom_x, right_bottom_y),
            (right_top_x, right_top_y),
        ]
    else:
        corners = [
            (left_top_x, left_top_y),
            (left_bottom_x, left_bottom_y),
            (right_bottom_x, right_bottom_y),
            (right_top_x, right_top_y),
        ]

    # draw_corners(corners,image_colored,is_left_page=True)
    cropped = crop_image(corners,image_colored)



    if not os.path.exists(r"C:\Users\user\PycharmProjects\pythonProject10\testings & mid Results\Cropped_R_S3"):
        os.makedirs(r"C:\Users\user\PycharmProjects\pythonProject10\testings & mid Results\Cropped_R_S3")
    if not os.path.exists(r"C:\Users\user\PycharmProjects\pythonProject10\testings & mid Results\Cropped_L_S3"):
        os.makedirs(r"C:\Users\user\PycharmProjects\pythonProject10\testings & mid Results\Cropped_L_S3")
    if is_left_page:
        file_path_right = os.path.join(r"C:\Users\user\PycharmProjects\pythonProject10\testings & mid Results\Cropped_L_S3"
                                       , f"cropped_L_{num}.png")
        cv2.imwrite(file_path_right, cropped)
    else:
        file_path_right = os.path.join(r"C:\Users\user\PycharmProjects\pythonProject10\testings & mid Results\Cropped_R_S3"
                                       , f"cropped_R_{num}.png")
        cv2.imwrite(file_path_right, cropped)


def analyze_all_crop(folder_path):
    for filename in sorted(os.listdir(folder_path)):
        image_path = os.path.join(folder_path, filename)
        if "left" in filename.lower():
            is_left_page = True
        elif "right" in filename.lower():
            is_left_page = False
            # חילוץ מספר עמוד מהשם
        match = re.search(r"_(\d+)", filename)
        num = match.group(1)
        analyze_page(image_path, is_left_page=is_left_page, num=num)

# ---------- הרצה ----------



# analyze_page(r"C:\Users\user\PycharmProjects\pythonProject10\Page_split_S2\", is_left_page=False)
# analyze_page(r"C:\Users\user\PycharmProjects\pythonProject10\left_page.jpg", is_left_page=True)
# analyze_all_crop(r"C:\Users\user\PycharmProjects\pythonProject10\Page_split_S2")