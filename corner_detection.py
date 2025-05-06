import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ---------- פונקציות עזר ----------

def find_y_from_right_to_left(binary, threshold_percent, from_top=True):
    height, width = binary.shape
    y_range = range(height) if from_top else range(height - 1, -1, -1)
    for y in y_range:
        row = binary[y, :]
        black_pixel_ratio = np.sum(row) / (255 * width) * 100
        if black_pixel_ratio <= threshold_percent:
            return y
    return -1

# לעמוד ימין: סריקה מימין לשמאל
def find_x_from_top_to_bottom(binary, threshold_percent):
    height, width = binary.shape
    for x in range(width - 1, -1, -1):  # ימין לשמאל
        col = binary[:, x]
        black_pixel_ratio = np.sum(col) / (255 * height) * 100
        if black_pixel_ratio <= threshold_percent:
            return x
    return -1

def find_x_from_bottom_to_top(binary, threshold_percent):
    height, width = binary.shape
    for x in range(width - 1, -1, -1):  # ימין לשמאל
        col = binary[:, x]
        black_pixel_ratio = np.sum(col) / (255 * height) * 100
        if black_pixel_ratio <= threshold_percent:
            return x
    return -1

# לעמוד שמאל: סריקה משמאל לימין
def find_x_from_top_to_bottom_left(binary, threshold_percent):
    height, width = binary.shape
    for x in range(width):  # משמאל לימין
        col = binary[:, x]
        black_pixel_ratio = np.sum(col) / (255 * height) * 100
        if black_pixel_ratio <= threshold_percent:
            return x
    return -1

def find_x_from_bottom_to_top_left(binary, threshold_percent):
    height, width = binary.shape
    for x in range(width):  # משמאל לימין
        col = binary[:, x]
        black_pixel_ratio = np.sum(col) / (255 * height) * 100
        if black_pixel_ratio <= threshold_percent:
            return x
    return -1


# ---------- פונקציית dewarp ----------

def four_point_transform(image, pts):
    rect = np.array(pts, dtype="float32")

    widthA = np.linalg.norm(rect[0] - rect[3])
    widthB = np.linalg.norm(rect[1] - rect[2])
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(rect[0] - rect[1])
    heightB = np.linalg.norm(rect[2] - rect[3])
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [0, maxHeight - 1],
        [maxWidth - 1, maxHeight - 1],
        [maxWidth - 1, 0]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


# ---------- ניתוח עמוד ----------

def analyze_page(image_path, is_left_page=False,num=""):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_colored = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Image not found! Check the file path.")

    image = cv2.GaussianBlur(image, (5, 5), 0.5)
    _, binary = cv2.threshold(image, 148, 255, cv2.THRESH_BINARY_INV)

    if is_left_page:
        # -------- עמוד שמאל --------
        right_bottom_x = binary.shape[1] - 1
        right_bottom_y = find_y_from_right_to_left(binary, 60, from_top=False)

        right_top_x = binary.shape[1] - 1
        right_top_y = find_y_from_right_to_left(binary, 60, from_top=True)

        left_top_x = find_x_from_top_to_bottom_left(binary, 60)
        left_top_y = find_y_from_right_to_left(binary, 60, from_top=True)

        left_bottom_x = find_x_from_bottom_to_top_left(binary, 60)
        left_bottom_y = find_y_from_right_to_left(binary, 60, from_top=False)

    else:
        # -------- עמוד ימין --------
        right_bottom_x = find_x_from_bottom_to_top(binary, 80)
        right_bottom_y = find_y_from_right_to_left(binary, 80, from_top=False)

        right_top_x = find_x_from_top_to_bottom(binary, 70)
        right_top_y = find_y_from_right_to_left(binary, 70, from_top=True)

        left_top_x = 0
        left_top_y = find_y_from_right_to_left(binary, 70, from_top=True)

        left_bottom_x = 0
        left_bottom_y = find_y_from_right_to_left(binary, 80, from_top=False)

    # ציור כל הפינות
    points = [
        ((right_top_x, right_top_y), (0, 255, 0), "Right Top"),
        ((right_bottom_x, right_bottom_y), (255, 0, 0), "Right Bottom"),
        ((left_bottom_x, left_bottom_y), (255, 255, 0), "Left Bottom"),
        ((left_top_x, left_top_y), (0, 0, 255), "Left Top"),
    ]

    for (x, y), color, label in points:
        if x != -1 and y != -1:
            # print(f"{label} Corner: ({x}, {y})")
            cv2.circle(image_colored, (x, y), 10, color, -1)
        # else:
            # print(f"{label} Corner: Not found")

    # תצוגת פינות
    # plt.figure(figsize=(12, 6))
    # title = "Left Page" if is_left_page else "Right Page"
    # plt.title(f"{title} - Detected Corners")
    # plt.imshow(cv2.cvtColor(image_colored, cv2.COLOR_BGR2RGB))
    # plt.axis("off")
    # plt.show()

    # יישור התמונה (Dewarp) – שינוי סדר הפינות לעמוד שמאל
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

    if all(c != -1 for pt in corners for c in pt):
        warped = four_point_transform(image_colored, corners)
        plt.figure(figsize=(10, 6))
        # plt.title(f"{title} - Dewarped")
        # plt.title(f" - Dewarped")
        # plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        # plt.axis("off")
        # plt.show()
        if not os.path.exists("selected_imagesB"):
            os.makedirs("selected_imagesB")

        # Save both pages
        file_path_right = os.path.join("selected_imagesB", f"dewarped_{num}")
        cv2.imwrite(file_path_right, warped)
    else:
        print("לא ניתן לבצע dewarp – לא נמצאו כל הפינות")



# ---------- הרצה ----------

# if __name__ == "__main__":
    # ניתוח עמוד ימין
# analyze_page(r"C:\Users\user\PycharmProjects\pythonProject10\selected_imagesA\right_page_selected_image_105.png", is_left_page=False)

    # ניתוח עמוד שמאל
    # analyze_page(r"C:\Users\user\PycharmProjects\pythonProject10\left_page.jpg", is_left_page=True)
