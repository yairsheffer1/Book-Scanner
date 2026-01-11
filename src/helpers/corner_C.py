import numpy as np
import cv2
from scipy.interpolate import splprep, splev
from shapely.geometry import LineString
import os
import matplotlib.pyplot as plt
import re

def find_edge_points(image, start_point, direction='horizontal', steps=25, reverse=False, from_right=False):
    points = []
    h, w = image.shape
    step = int(max(1, 0.01 * (w * 2 if direction == 'horizontal' else h)))

    for i in range(steps + 2):
        delta = -i * step

        if direction == 'horizontal':
            y = int(start_point[1] - delta)
            if y < 0 or y >= h:
                continue
            x = find_first_white_x(image, y, from_right=not reverse)
            if x != -1:
                points.append((x, y))
        else:
            x = int(start_point[0] + delta)
            if x < 0 or x >= w:
                continue
            y = find_first_white_y(image, x)
            if y != -1:
                points.append((x, y))

    return points

def find_first_white_x(image, y_start, min_run=10, from_right=True):
    height, width = image.shape
    step = int(0.03 * height)

    for y in range(y_start, height, step):
        count = 0
        if from_right:
            x_range = range(width - 1, -1, -1)
        else:
            x_range = range(width)  # סריקה משמאל לימין

        for x in x_range:
            if image[y, x] == 0:
                count += 1
                if count >= min_run:
                    return x
            else:
                count = 0
    return -1

def find_first_white_y(image, x, min_run=5, from_bottom=True):
    count = 0
    if from_bottom:
        y_range = range(image.shape[0] - 1, -1, -1)
    else:
        y_range = range(image.shape[0])  # סריקה מלמעלה למטה

    for y in y_range:
        if image[y, x] == 0:
            count += 1
            if count >= min_run:
                return y
        else:
            count = 0
    return -1

def fit_spline(points):
    points = np.array(points)
    if len(points) < 3:
        return None
    tck, _ = splprep([points[:, 0], points[:, 1]], s=15)
    return tck

def evaluate_spline(tck, num=100, extend_ratio=1.5):
    u = np.linspace(0, 1 + extend_ratio, num)
    out = splev(u, tck)
    return np.vstack(out).T

def find_intersection(curve1, curve2):
    line1 = LineString(curve1)
    line2 = LineString(curve2)
    intersection = line1.intersection(line2)

    if intersection.is_empty:
        return None
    elif intersection.geom_type == 'Point':
        return np.array(intersection.coords[0])
    elif intersection.geom_type == 'MultiPoint':
        return np.array(intersection.geoms[0].coords[0])
    else:
        return None

def line_slope(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dx == 0:
        return np.inf  # קו אנכי
    return dy / dx

def line_from_point_slope(point, slope, x_vals):
    x0, y0 = point
    if slope == np.inf:
        # קו אנכי
        return np.column_stack((np.full_like(x_vals, x0), x_vals))
    y_vals = slope * (x_vals - x0) + y0
    return np.column_stack((x_vals, y_vals))

def find_intersection_line_curve(line_pts, curve_pts):
    line = LineString(line_pts)
    curve = LineString(curve_pts)
    inter = line.intersection(curve)
    if inter.is_empty:
        return None
    if inter.geom_type == 'Point':
        return np.array(inter.coords[0])
    if inter.geom_type == 'MultiPoint':
        return np.array(inter.geoms[0].coords[0])
    return None

def hough_lowest_vertical_line(image):
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
    if lines is None:
        return None

    lowest_line = None
    max_y = -1

    for rho, theta in lines[:, 0]:
        angle_deg = theta * 180 / np.pi
        if 80 <= angle_deg <= 100:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho

            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            line_max_y = max(y1, y2)
            if line_max_y > max_y:
                max_y = line_max_y
                lowest_line = ((x1, y1), (x2, y2))

    return lowest_line

def process_with_hough(binary_image, P1, P2, P3, is_left=False, output_path=''):

    height, width = binary_image.shape
    bottom_edge_points = find_edge_points(binary_image, P2, direction='horizontal', reverse=is_left)[1:]
    tck_bottom = fit_spline(bottom_edge_points)

    if tck_bottom is None:
        # print("לא נמצאו מספיק נקודות לקו התחתון")
        return None

    bottom_curve = evaluate_spline(tck_bottom, extend_ratio=3)

    lowest_hough_line = hough_lowest_vertical_line(binary_image)
    if lowest_hough_line is None:
        # print("לא נמצא קו Hough אנכי")
        return None

    slope = line_slope(lowest_hough_line[0], lowest_hough_line[1])
    # print(f"שיפוע קו ה-Hough: {slope}")

    new_x = int(width * 0.03)
    new_y = find_first_white_y(binary_image, new_x)
    if new_y == -1:
        # print("לא נמצאה נקודה מתאימה לפי find_first_white_y, משתמש ב-P3")
        new_point = P3
    else:
        new_point = (new_x, new_y)
    # print(f"נקודת התחלה חדשה של הקו: {new_point}")

    x_min = 0
    x_max = width
    line_pts = line_from_point_slope(new_point, slope, np.linspace(x_min, x_max, 500))

    intersection_point = find_intersection_line_curve(line_pts, bottom_curve)

    image_color = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

    bottom_curve_pts = np.array(bottom_curve, dtype=np.int32)
    cv2.polylines(image_color, [bottom_curve_pts], isClosed=False, color=(0, 100, 255), thickness=10)

    line_pts_int = np.array(line_pts, dtype=np.int32)
    cv2.polylines(image_color, [line_pts_int], isClosed=False, color=(255, 0, 0), thickness=3)

    for pt, color in zip([P1, P2, P3, new_point], [(255, 255, 0), (0, 255, 255), (255, 0, 255), (0, 255, 0)]):
        cv2.circle(image_color, (int(pt[0]), int(pt[1])), 12, color, -1)

    if intersection_point is not None:
        x, y = int(round(intersection_point[0])), int(round(intersection_point[1]))
        cv2.circle(image_color, (x, y), 14, (0, 0, 255), -1)
        print(f"נקודת מפגש: {intersection_point}")
    else:
        print("לא נמצאה נקודת מפגש בין הקווים")

    show_image_matplotlib(image_color, title="Hough + Curve Intersection")

    cv2.imwrite(output_path, image_color)
    print(f"תמונה נשמרה ב: {output_path}")

    return intersection_point

# ----- פינות (לא לשנות) -----
def R_get_right_top_y(binary):
    height, width = binary.shape
    for y in range(height):
        row = binary[y, :]
        black_pixel_ratio = np.sum(row) / (255 * width)
        if black_pixel_ratio <= 0.8:
            y_start = y
            break
    else:
        return -1

    y_current = y_start
    ratio_current = black_pixel_ratio
    step = int(0.01 * height)

    while y_current + step < height:
        y_next = y_current + step
        row_next = binary[y_next, :]
        ratio_next = np.sum(row_next) / (255 * width)
        if abs(ratio_next - ratio_current) >= 0.06:
            y_current = y_next
            ratio_current = ratio_next
        else:
            break

    return y_current

def R_get_right_top_x(binary, y):
    height, width = binary.shape

    def has_black_pixel_sequence(binary, x, y, required_black=5, window=7):
        black_count = 0
        start_x = max(0, x - window + 1)
        for i in range(x, start_x - 1, -1):
            if binary[(y + int(0.01 * height)), i] == 0:
                black_count += 1
        return black_count >= required_black

    for x in range(width - 1, -1, -1):
        col = binary[:, x]
        black_pixel_ratio = np.sum(col) / (255 * height)
        if black_pixel_ratio <= 0.8:
            if has_black_pixel_sequence(binary, x, y):
                return x
    return -1

def R_get_top_left_y(binary):
    height, width = binary.shape

    def check_initial_white(row, needed=3, total=11):
        blacks = np.sum(row[:total] == 0)
        return blacks >= needed

    for y in range(height):
        row = binary[y, :]
        black_pixel_ratio = np.sum(row) / (255 * width)
        if black_pixel_ratio <= 0.8 and check_initial_white(row):
            return y
    return -1

def R_get_bottom_left_y(binary):
    height, width = binary.shape

    def check_initial_white(row, needed=9, total=11):
        blacks = np.sum(row[:total] == 0)
        return blacks >= needed

    for y in range(height - 1, -1, -1):
        row = binary[y, :]
        black_pixel_ratio = np.sum(row) / (255 * width)
        if black_pixel_ratio <= 0.8 and check_initial_white(row):
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





def order_points(pts):
    # סדר הנקודות: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # חישוב רוחב חדש
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    # חישוב גובה חדש
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    # נקודות יעד
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # מטריצת הטרנספורמציה ויישור
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def straighten_page(image, P1, P2, P3, P4, output_path=None, image_path=None, is_left_page=False, num=""):
    """
    מיישרת את הדף באמצעות 4 נקודות פינה. אם אחת חסרה, מפעילה את analyze_page.
    """

    # בדיקה האם כל הפינות קיימות
    if any(p is None for p in [P1, P2, P3, P4]):
        if image_path is None:
            raise ValueError("image_path נדרש אם אחת הנקודות חסרה ויש להפעיל analyze_page")
        print("⛔ לא נמצאו כל הפינות - מופעלת analyze_page.")
        # return analyze_page(image_path, is_left_page, num)

    # ודא שהתמונה בצבע
    if len(image.shape) == 2:
        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_color = image.copy()

    # ציור הפינות
    for point, color in zip([P1, P2, P4, P3], [(0, 0, 255), (0, 255, 0), (255, 0, 255), (255, 255, 0)]):
        if point is not None:
            cv2.circle(image_color, (int(point[0]), int(point[1])), 14, color, -1)

    # יישור מבוסס חישוב דינמי
    corners = np.array([P1, P2, P4, P3], dtype="float32")
    warped = four_point_transform(image, corners)

    # שמירה אם נתבקש
    if output_path:
        cv2.imwrite(output_path, warped)
        print(f"✅ תמונה מיושרת נשמרה: {output_path}")

    return warped


def show_image_matplotlib(image, title="Image", figsize=(12, 6)):
    """
    מציג תמונה בפורמט BGR עם matplotlib.
    """
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()




# ---------- פונקציית זיהוי צד לפי שם ----------
def detect_side_from_filename(filename):
    lower = filename.lower()
    if "left" in lower or "_l" in lower:
        return "left"
    elif "right" in lower or "_r" in lower:
        return "right"
    else:
        return "unknown"

# ---------- תהליך עיבוד התיקייה עם זיהוי צד ----------
def process_folder(folder_path):
    for filename in sorted(os.listdir(folder_path)):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        image_path = os.path.join(folder_path, filename)
        print(f"\nמעבד קובץ: {filename}")
        match = re.search(r"_(\d+)", filename)
        num = match.group(1)

        image_color = cv2.imread(image_path)
        if image_color is None:
            print("לא ניתן לקרוא את התמונה:", image_path)
            continue

        image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

        _, binary0 = cv2.threshold(image_gray, 170, 255, cv2.THRESH_BINARY_INV)
        _, binary1 = cv2.threshold(image_gray, 200, 255, cv2.THRESH_BINARY_INV)

        side = detect_side_from_filename(filename)
        if side == "right":
            right_top_y = R_get_right_top_y(binary1)
            right_top_x = R_get_right_top_x(binary1, right_top_y)
            left_top_x = 0
            left_top_y = R_get_top_left_y(binary0)
            left_bottom_x = 0
            left_bottom_y = R_get_bottom_left_y(binary0)
            P1 = (left_top_x, left_top_y)
            P2 = (right_top_x, right_top_y)
            P3 = (left_bottom_x, left_bottom_y)
            P4 = process_with_hough(binary1, P1, P2, P3,False)

        elif side == "left":
            right_top_x = image_color.shape[1] - 1
            right_top_y = L_get_top_right_y(binary0)
            left_top_y = L_get_left_top_y(binary1)
            left_top_x = L_get_left_top_x(binary1, left_top_y)
            right_bottom_x = image_color.shape[1] - 1
            right_bottom_y = L_get_bottom_right_y(binary0)
            P1 = (left_top_x, left_top_y)
            P2 = (right_top_x, right_top_y)
            P4 = (right_bottom_x, right_bottom_y)
            P3 = process_with_hough(binary1, P1, P2, P4,True)

        else:
            print("לא ניתן לזהות צד לפי שם הקובץ, מדלג...")
            continue


        straightened = straighten_page(image_color, P1, P2, P3, P4,None,image_path,side=="left")
        show_image_matplotlib(straightened, title=f"Straightened - {filename}")

        if not os.path.exists("../../testings & mid Results/Dewarped_R_S3"):
            os.makedirs("../../testings & mid Results/Dewarped_R_S3")
        if not os.path.exists("../../testings & mid Results/Dewarped_L_S3"):
            os.makedirs("../../testings & mid Results/Dewarped_L_S3")
        if side == "left":
            file_path_right = os.path.join("../../testings & mid Results/Dewarped_L_S3", f"dewarped_L_{num}.png")
            cv2.imwrite(file_path_right, straightened)
        else:
            file_path_right = os.path.join("../../testings & mid Results/Dewarped_R_S3", f"dewarped_R_{num}.png")
            cv2.imwrite(file_path_right, straightened)


        user_input = input("לחץ Enter להמשך, או 'q' כדי לצאת: ")
        if user_input.lower() == 'q':
            print("עוצר את העיבוד.")
            break

# -------- הפעלה --------
if __name__ == "__main__":
    folder = r"C:\Users\user\PycharmProjects\pythonProject10\page_split_S2"
    process_folder(folder)

# # --- טעינת תמונה והרצה ---
# if __name__ == "__main__":
#     image_path = r"C:\Users\user\PycharmProjects\pythonProject10\finals_R_S4\ropped_R_190.png"
#     image_color = cv2.imread(image_path)
#     image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
#
#     # תמשיך עם שאר הקוד עם image_gray
#
#     _, binary0 = cv2.threshold(image, 170, 255, cv2.THRESH_BINARY_INV)
#     _, binary1 = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)
#
#     right_top_y = R_get_right_top_y(binary1)
#     right_top_x = R_get_right_top_x(binary1, right_top_y)
#
#     left_top_x = 0
#     left_top_y = R_get_top_left_y(binary0)
#     left_bottom_x = 0
#     left_bottom_y = R_get_bottom_left_y(binary0)
#
#     P1 = (left_top_x, left_top_y)
#     P2 = (right_top_x, right_top_y)
#     P3 = (left_bottom_x, left_bottom_y)
#
#     P4 = process_with_hough(binary1, P1, P2, P3)
#     straighten_page(
#         image_color, P1, P2, P3, P4,

#     )
#
