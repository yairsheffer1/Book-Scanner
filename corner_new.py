import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ---------- פונקציות עזר ----------

def find_y_from_right_to_left(binary, threshold_percent, from_top=True, check_start=False):
    height, width = binary.shape
    y_range = range(height) if from_top else range(height - 1, -1, -1)

    def check_initial_black(row, needed=9, total=11):
        blacks = np.sum(row[:total] == 255)
        return blacks >= needed

    for y in y_range:
        row = binary[y, :]
        black_pixel_ratio = np.sum(row) / (255 * width) * 100
        if black_pixel_ratio <= threshold_percent:
            if check_start:
                if check_initial_black(row):
                    return y
            else:
                return y
    return -1

def find_y_from_left_to_right(binary, threshold_percent, from_top=True, check_start=False):
    height, width = binary.shape
    y_range = range(height) if from_top else range(height - 1, -1, -1)

    def check_initial_white(row, needed=9, total=11):
        blacks = np.sum(row[:total] == 0)
        return blacks >= needed

    for y in y_range:
        row = binary[y, :]
        black_pixel_ratio = np.sum(row) / (255 * width) * 100
        if black_pixel_ratio <= threshold_percent:
            if check_start:
                if check_initial_white(row):
                    return y
            else:
                return y
    return -1


def find_x_from_top_to_bottom(binary, threshold_percent, y,check_start=False):
    height, width = binary.shape

    def check_initial_black(col, needed=9, total=11):
        blacks = np.sum(col[:total] == 255)
        return blacks >= needed

    def verify_black_patch(binary, x, y, patch_size=5, threshold_black_ratio=0.9):
        height, width = binary.shape
        offset_y = int(0.1 * height)  # 10% מהגובה
        new_y = y + offset_y
        # if new_y + patch_size >= height or x + patch_size >= width:
        #     return False  # חורג מגבולות התמונה

        patch = binary[new_y:new_y + patch_size, x:x + patch_size]
        black_pixels = np.sum(patch == 255)
        total_pixels = patch_size * patch_size
        return black_pixels / total_pixels >= threshold_black_ratio

    def has_black_pixel_sequence(binary, x, y, required_black=5, window=7):
        """
        בודק אם בשורה y, מ-x שמאלה, יש לפחות `required_black` פיקסלים שחורים מתוך `window` פיקסלים.
        """
        height, width = binary.shape
        black_count = 0

        start_x = max(0, x - window + 1)
        for i in range(x, start_x - 1, -1):  # הולכים שמאלה
            if binary[y, i] == 0:  # פיקסל שחור
                black_count += 1

        return black_count >= required_black

    for x in range(width - 1, -1, -1):  # ימין לשמאל
        col = binary[:, x]
        black_pixel_ratio = np.sum(col) / (255 * height) * 100
        if black_pixel_ratio <= threshold_percent:
            if check_start:
                if check_initial_black(col) and verify_black_patch(binary,x,y) and has_black_pixel_sequence(binary,x,y):
                    return x
            else:
                if check_initial_black(col) and verify_black_patch(binary, x, y) and has_black_pixel_sequence(binary,x, y):
                    return x
    return -1

def find_x_from_bottom_to_top(binary, threshold_percent, check_start=False):
    height, width = binary.shape

    def check_initial_black(col, needed=9, total=11):
        blacks = np.sum(col[:total] == 255)
        return blacks >= needed

    for x in range(width - 1, -1, -1):  # ימין לשמאל
        col = binary[:, x]
        black_pixel_ratio = np.sum(col) / (255 * height) * 100
        if black_pixel_ratio <= threshold_percent:
            if check_start:
                if check_initial_black(col):
                    return x
            else:
                return x
    return -1

def find_x_from_top_to_bottom_left(binary, threshold_percent, check_start=False):
    height, width = binary.shape

    def check_initial_black(col, needed=9, total=11):
        blacks = np.sum(col[:total] == 255)
        return blacks >= needed

    for x in range(width):  # משמאל לימין
        col = binary[:, x]
        black_pixel_ratio = np.sum(col) / (255 * height) * 100
        if black_pixel_ratio <= threshold_percent:
            if check_start:
                if check_initial_black(col):
                    return x
            else:
                return x
    return -1

def find_x_from_bottom_to_top_left(binary, threshold_percent, check_start=False):
    height, width = binary.shape

    def check_initial_black(col, needed=9, total=11):
        blacks = np.sum(col[:total] == 255)
        return blacks >= needed

    for x in range(width):  # משמאל לימין
        col = binary[:, x]
        black_pixel_ratio = np.sum(col) / (255 * height) * 100
        if black_pixel_ratio <= threshold_percent:
            if check_start:
                if check_initial_black(col):
                    return x
            else:
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

def analyze_page(image_path, is_left_page=False, num=""):
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

        right_top_y = find_y_from_right_to_left(binary, 70, from_top=True, check_start=False)
        right_top_x = find_x_from_top_to_bottom(binary, 70, right_top_y,check_start=False )

        left_top_x = 0
        left_top_y = find_y_from_left_to_right(binary, 70, from_top=True, check_start=True)

        left_bottom_x = 0
        left_bottom_y = find_y_from_right_to_left(binary, 80, from_top=False, check_start=False)

    # ציור כל הפינות
    points = [
        ((right_top_x, right_top_y), (0, 255, 0), "Right Top"),
        ((right_bottom_x, right_bottom_y), (255, 0, 0), "Right Bottom"),
        ((left_bottom_x, left_bottom_y), (255, 255, 0), "Left Bottom"),
        ((left_top_x, left_top_y), (0, 0, 255), "Left Top"),
    ]

    for (x, y), color, label in points:
        if x != -1 and y != -1:
            print(f"{label} Corner: ({x}, {y})")
            cv2.circle(image_colored, (x, y), 10, color, -1)
        else:
            print(f"{label} Corner: Not found")

    # יישור התמונה (Dewarp)
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
        plt.title(f"Dewarped Page")
        plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

        if not os.path.exists("selected_imagesB"):
            os.makedirs("selected_imagesB")

        file_path_right = os.path.join("selected_imagesB", f"dewarped_{num}")
        # כאן תוכל לשמור את הקובץ אם תרצה:
        # cv2.imwrite(file_path_right, warped)

    else:
        print("לא ניתן לבצע dewarp – לא נמצאו כל הפינות")

# ---------- הרצה לדוגמה ----------

# אם אתה רוצה להריץ:
# analyze_page(r"נתיב_לקובץ", is_left_page=False)

analyze_page(r"C:\Users\user\PycharmProjects\pythonProject10\selected_imagesA\right_page_selected_image_105.png", is_left_page=False)

    # ניתוח עמוד שמאל
    # analyze_page(r"C:\Users\user\PycharmProjects\pythonProject10\left_page.jpg", is_left_page=True)
