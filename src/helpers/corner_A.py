import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import re

# ---------- פונקציות עזר ----------

def find_y_with_vertical_white_run(binary, min_white_run=5, x_start_ratio=0.05, from_top=False):
    height, width = binary.shape
    x_start = int(width * x_start_ratio)

    # עבור כל x החל מ-x_start ועד סוף התמונה
    for x in range(x_start, width):
        white_run = 0
        y_range = range(height) if from_top else range(height - 1, -1, -1)

        for y in y_range:
            if binary[y, x] == 0:
                white_run += 1
                if white_run >= min_white_run:
                    return y  # מחזיר את y שבו נמצא סוף הרצף
            else:
                white_run = 0

    return -1  # לא נמצא רצף מתאים


def R_get_right_bottom(binary, y_top):
    height, width = binary.shape

    bottum_line_y2 = find_y_with_vertical_white_run(binary,5, 0.1)
    bottum_line_y1 = find_y_with_vertical_white_run(binary, 5, 0.05)
    bottum_line_x2 = int(width * 0.1)
    bottum_line_x1 = int(width * 0.05)

    offset_5 = int(0.1 * height)
    offset_15 = int(0.15 * height)
    y1 = min(y_top + offset_5, height - 1)
    y2 = min(y_top + offset_15, height - 1)

    def find_right_edge_x(y):
        count = 0
        for x in range(width - 1, -1, -1):
            if binary[y, x] == 0:
                count += 1
                if count >= 5:
                    return x
            else:
                count = 0
        return -1

    x1 = find_right_edge_x(y1)
    x2 = find_right_edge_x(y2)

    # 4 נקודות:
    p1 = (bottum_line_x1, bottum_line_y1)
    p2 = (bottum_line_x2, bottum_line_y2)
    p3 = (x1, y1)
    p4 = (x2, y2)

    intersection = find_intersection(p1, p2, p3, p4)
    return intersection

def find_intersection(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
    if denom == 0:
        return None  # קווים מקבילים

    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom

    return int(px), int(py)

def L_get_left_bottom(binary, y_top):
    height, width = binary.shape

    # שורות שבהן יש רצף לבן אנכי (קו תחתון)
    bottum_line_y2 = find_y_with_vertical_white_run(binary, 5, 0.1)  # x ≈ 10%
    bottum_line_y1 = find_y_with_vertical_white_run(binary, 5, 0.05)  # x ≈ 5%
    bottum_line_x2 = width - int(width * 0.1)  # היפוך X
    bottum_line_x1 = width - int(width * 0.05)

    # זזים כלפי מטה (Y רגיל)
    offset_5 = int(0.1 * height)
    offset_15 = int(0.15 * height)
    y1 = min(y_top + offset_5, height - 1)
    y2 = min(y_top + offset_15, height - 1)

    # מחפשים את הקצה ה"שחור" מצד שמאל של הדף (כלומר מימין של התמונה)
    def find_right_edge_x_flipped(y):
        count = 0
        for x in range(width):  # מימין לשמאל בהקשר ל"דף"
            if binary[y, x] == 0:
                count += 1
                if count >= 5:
                    return x
            else:
                count = 0
        return -1

    x1 = find_right_edge_x_flipped(y1)
    x2 = find_right_edge_x_flipped(y2)

    # ארבע נקודות: קו תחתון וקו צד
    p1 = (bottum_line_x1, bottum_line_y1)
    p2 = (bottum_line_x2, bottum_line_y2)
    p3 = (x1, y1)
    p4 = (x2, y2)

    intersection = find_intersection(p1, p2, p3, p4)
    return intersection

def find_intersection(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
    if denom == 0:
        return None  # קווים מקבילים

    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom

    return int(px), int(py)



def R_get_right_top_y(binary):
    height, width = binary.shape
    y_range = range(height)


    for y in y_range:
        row = binary[y, :]
        black_pixel_ratio = np.sum(row) / (255 * width)
        if black_pixel_ratio <= 0.8:
            y_start = y
            break
    else:
        return -1  # אם לא נמצאה נקודה מתאימה

    # המשך בדיקה בשני אחוז כל פעם
    y_current = y_start
    ratio_current = black_pixel_ratio
    step = int(0.01 * height)

    while y_current + step < height:
        y_next = y_current + step
        row_next = binary[y_next, :]
        ratio_next = np.sum(row_next) / (255 * width)

        # אם אחוז הפיקסלים השחורים ירד ב-5% או יותר
        if abs(ratio_next - ratio_current) >= 0.06:
            y_current = y_next
            ratio_current = ratio_next
        else:
            break

    return y_current

def R_get_right_top_x(binary,y):
    height, width = binary.shape

    def has_black_pixel_sequence(binary, x, y, required_black=5, window=7):
        height, width = binary.shape
        black_count = 0

        start_x = max(0, x - window + 1)
        for i in range(x, start_x - 1, -1):  # הולכים שמאלה
            if binary[(y + int(0.01 * height)), i] == 0:  # פיקסל שחור
                black_count += 1

        return black_count >= required_black

    for x in range(width - 1, -1, -1):  # ימין לשמאל
        col = binary[:, x]
        black_pixel_ratio = np.sum(col) / (255 * height)
        if black_pixel_ratio <= 0.8:
            if has_black_pixel_sequence(binary, x,y):
                return x
    return -1

def R_get_top_left_y(binary):
    height, width = binary.shape
    y_range = range(height)

    def check_initial_white(row, needed=3, total=11):
        blacks = np.sum(row[:total] == 0)
        return blacks >= needed

    for y in y_range:
        row = binary[y, :]
        black_pixel_ratio = np.sum(row) / (255 * width)
        if black_pixel_ratio <= 0.8:
            if check_initial_white(row):
                return y
    return -1

def R_get_bottom_left_y(binary):
    height, width = binary.shape
    y_range = range(height - 1, -1, -1)

    def check_initial_white(row, needed=9, total=11):
        blacks = np.sum(row[:total] == 0)
        return blacks >= needed



    for y in y_range:
        row = binary[y, :]
        black_pixel_ratio = np.sum(row) / (255 * width)
        if black_pixel_ratio <= 0.8:
            if check_initial_white(row):
                return y
    return -1




def L_get_bottom_right_y(binary):
    height, width = binary.shape
    y_range = range(height - 1, -1, -1)

    def check_initial_white(row, needed=9, total=11):
        blacks = np.sum(row[binary.shape[1] - 1 - total:binary.shape[1] - 1] == 0)
        return blacks >= needed

    for y in y_range:
        row = binary[y, :]
        black_pixel_ratio = np.sum(row) / (255 * width)
        if black_pixel_ratio <= 0.8:
            if check_initial_white(row):
                return y
    return -1

def L_get_top_right_y(binary):
    height, width = binary.shape
    y_range = range(height)

    def check_initial_white(row, needed=3, total=11):
        blacks = np.sum(row[binary.shape[1] - 1 - total:binary.shape[1] - 1] == 0)
        return blacks >= needed

    for y in y_range:
        row = binary[y, :]
        black_pixel_ratio = np.sum(row) / (255 * width)
        if black_pixel_ratio <= 0.8:
            if check_initial_white(row):
                return y
    return -1

def L_get_left_top_y(binary):
    height, width = binary.shape
    y_range = range(height)

    for y in y_range:
        row = binary[y, :]
        black_pixel_ratio = np.sum(row) / (255 * width)
        if black_pixel_ratio <= 0.8:
            y_start = y
            break
    else:
        return -1  # אם לא נמצאה נקודה מתאימה

    # המשך בדיקה בשני אחוז כל פעם
    y_current = y_start
    ratio_current = black_pixel_ratio
    step = int(0.01 * height)

    while y_current + step < height:
        y_next = y_current + step
        row_next = binary[y_next, :]
        ratio_next = np.sum(row_next) / (255 * width)

        # אם אחוז הפיקסלים השחורים ירד ב-5% או יותר
        if abs(ratio_next - ratio_current) >= 0.06:
            y_current = y_next
            ratio_current = ratio_next
        else:
            break

    return y_current

def L_get_left_top_x(binary, y):
    height, width = binary.shape

    def has_black_pixel_sequence(binary, x, y, required_black=5, window=7):
        height, width = binary.shape
        black_count = 0

        end_x = min(width, x + window)
        for i in range(x, end_x):  # הולכים ימינה
            if binary[(y + int(0.01 * height)), i] == 0:  # פיקסל שחור
                black_count += 1

        return black_count >= required_black

    for x in range(0, width):  # שמאל לימין
        col = binary[:, x]
        black_pixel_ratio = np.sum(col) / (255 * height)
        if black_pixel_ratio <= 0.8:
            if has_black_pixel_sequence(binary, x, y):
                return x
    return -1

def L_get_left_bottom_x(binary, y_top, y_bottom):
    height, width = binary.shape

    # אחוזים מתחת ל־y_top (כמו בימין)
    offset_5 = int(0.05 * height)
    offset_15 = int(0.15 * height)

    y1 = min(y_top + offset_5, height - 1)
    y2 = min(y_top + offset_15, height - 1)

    # חיפוש ערך ה־x השמאלי ביותר בכל שורה (מפיקסלים שחורים)
    def find_left_edge_x(y):
        for x in range(width):
            if binary[y, x] == 0:
                return x
        return -1

    x1 = find_left_edge_x(y1)
    x2 = find_left_edge_x(y2)

    if x1 == -1 or x2 == -1:
        return -1

    delta_y = y2 - y1
    delta_x = x2 - x1

    if delta_y == 0:
        return x2

    slope = delta_x / delta_y
    y_target = y_bottom - y1
    estimated_x = int(x1 + slope * y_target)

    return estimated_x

def L_get_left_bottom_y(binary):
    height, width = binary.shape
    y_range = range(height - 1, -1, -1)  # מלמטה למעלה

    for y in y_range:
        row = binary[y, :]
        black_pixel_ratio = np.sum(row) / (255 * width)
        if black_pixel_ratio <= 0.7:
            y_start = y
            break
    else:
        return -1  # אם לא נמצאה נקודה מתאימה

    # המשך בדיקה בשני אחוז כל פעם (אבל כלפי מעלה)
    y_current = y_start
    ratio_current = black_pixel_ratio
    step = int(0.01 * height)

    while y_current - step >= 0:
        y_next = y_current - step
        row_next = binary[y_next, :]
        ratio_next = np.sum(row_next) / (255 * width)

        # אם אחוז הפיקסלים השחורים ירד ב-6% או יותר
        if abs(ratio_next - ratio_current) >= 0.06:
            y_current = y_next
            ratio_current = ratio_next
        else:
            break

    return y_current




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
    # _, binary = cv2.threshold(image, 170, 255, cv2.THRESH_BINARY_INV)
    # plt.figure(figsize=(14, 8))
    # plt.imshow(cv2.cvtColor(binary, cv2.COLOR_BGR2RGB))
    # plt.title("Original (Left) vs Dewarped (Right)")
    # plt.axis("off")
    # plt.show()

    _, binary0 = cv2.threshold(image, 170, 255, cv2.THRESH_BINARY_INV)
    _, binary1 = cv2.threshold(image, 190, 255, cv2.THRESH_BINARY_INV)
    if is_left_page:
        # -------- עמוד שמאל --------
        right_bottom_x = binary0.shape[1] - 1
        right_bottom_y = L_get_bottom_right_y(binary0)

        right_top_x = binary0.shape[1] - 1
        right_top_y = L_get_top_right_y(binary0)

        left_top_y = L_get_left_top_y(binary1)
        left_top_x = L_get_left_top_x(binary1, left_top_y)

        # left_bottom_y = L_get_left_bottom_y(binary)
        left_bottom_x, left_bottom_y = L_get_left_bottom(binary1, left_top_y)

    else:
        # -------- עמוד ימין --------
        right_top_y = R_get_right_top_y(binary1)
        right_top_x = R_get_right_top_x(binary1 ,right_top_y)

        left_top_x = 0
        left_top_y = R_get_top_left_y(binary0)

        left_bottom_x = 0
        left_bottom_y = R_get_bottom_left_y(binary0)

        right_bottom_x,right_bottom_y = R_get_right_bottom(binary1,right_top_y)


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

        # התאמת גובה התמונות אם נדרש
        h1, w1 = image_colored.shape[:2]
        h2, w2 = warped.shape[:2]
        if h1 != h2:
            scale = h1 / h2
            warped = cv2.resize(warped, (int(w2 * scale), h1))

        combined = np.hstack((image_colored, warped))

        # הצגת התמונה המאוחדת
        plt.figure(figsize=(14, 8))
        plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
        plt.title("Original (Left) vs Dewarped (Right)")
        plt.axis("off")
        plt.show()

        if not os.path.exists("../../testings & mid Results/Dewarped_R_S3"):
            os.makedirs("../../testings & mid Results/Dewarped_R_S3")
        if not os.path.exists("../../testings & mid Results/Dewarped_L_S3"):
            os.makedirs("../../testings & mid Results/Dewarped_L_S3")
        if is_left_page:
            file_path_right = os.path.join("../../testings & mid Results/Dewarped_L_S3", f"dewarped_L_{num}.png")
            cv2.imwrite(file_path_right, warped)
        else:
            file_path_right = os.path.join("../../testings & mid Results/Dewarped_R_S3", f"dewarped_R_{num}.png")
            cv2.imwrite(file_path_right, warped)

    else:
        print("לא ניתן לבצע dewarp – לא נמצאו כל הפינות")


def analze_all(path):
    # לולאה על כל הקבצים בתיקייה
    for filename in sorted(os.listdir(path)):

        image_path = os.path.join(path, filename)

        if "left" in filename.lower():
            is_left_page = True
        elif "right" in filename.lower():
            is_left_page = False

        # חילוץ המספר מהשם באמצעות regex
        match = re.search(r"_(\d+)", filename)
        num = match.group(1)
        analyze_page(image_path, is_left_page=is_left_page, num=num)

# analze_all(r"C:\Users\user\PycharmProjects\pythonProject10\Page_split_S2")
# ---------- הרצה לדוגמה ----------

# analyze_page(r"C:\Users\user\PycharmProjects\pythonProject10\Page_split_S2\right_page_165.png", is_left_page=False)

