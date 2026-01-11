import numpy as np
import cv2
from scipy.interpolate import splprep, splev
from shapely.geometry import LineString


def find_edge_points(image, start_point, direction='horizontal', steps=30, reverse=False):
    points = []
    h, w = image.shape
    step = int(max(1, 0.01 * (w * 2 if direction == 'horizontal' else h)))

    for i in range(steps + 2):
        delta = -i * step if reverse else i * step

        if direction == 'horizontal':
            y = int(start_point[1] - delta)
            if y < 0 or y >= h:
                continue
            x = find_first_white_x(image, y)
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


def find_first_white_x(image, y_start, min_run=10):
    height, width = image.shape
    step = int(0.03 * height)

    for y in range(y_start, height, step):
        count = 0
        for x in range(width - 1, -1, -1):
            if image[y, x] == 0:
                count += 1
                if count >= min_run:
                    return x
            else:
                count = 0
    return -1


def find_first_white_y(image, x):
    count = 0
    for y in range(image.shape[0] - 1, -1, -1):
        if image[y, x] == 0:
            count += 1
            if count >= 5:
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


def process(image, P1, P2, P3):
    right_edge_points = find_edge_points(image, P3, direction='vertical')[1:]
    bottom_edge_points = find_edge_points(image, P2, direction='horizontal', reverse=True)[1:]

    tck_right = fit_spline(right_edge_points)
    tck_bottom = fit_spline(bottom_edge_points)

    if tck_right is None or tck_bottom is None:
        print("לא נמצאו מספיק נקודות לקימור")
        return None

    right_curve = evaluate_spline(tck_right, extend_ratio=1.5)
    bottom_curve = evaluate_spline(tck_bottom, extend_ratio=1.5)
    # העתק לציור נקודות זווית על תמונה צבעונית
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    P4 = find_intersection(right_curve, bottom_curve)
    if P4 is not None:
        print("הפינה הרביעית:", P4)
    else:
        print("לא נמצאה הצטלבות בין העקומות")

    # ציור נקודות שוליים על התמונה הצבעונית
    for pt in right_edge_points:
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(image_color, (x, y), 10, (255, 0, 0), -1)  # כחול
    for pt in bottom_edge_points:
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(image_color, (x, y), 10, (0, 255, 0), -1)  # ירוק

    # ציור הפינות
    for pt, color in zip([P1, P2, P3], [(255, 255, 0), (0, 255, 255), (255, 0, 255)]):
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(image_color, (x, y), 12, color, -1)
    if P4 is not None:
        x, y = int(round(P4[0])), int(round(P4[1]))
        cv2.circle(image_color, (x, y), 14, (0, 0, 255), -1)  # אדום

    # ציור עקומות (קו מחבר נקודות splines)
    right_curve_pts = np.array(right_curve, dtype=np.int32)
    bottom_curve_pts = np.array(bottom_curve, dtype=np.int32)

    # cv2.polylines(image_color, [right_curve_pts], isClosed=False, color=(255, 0, 255), thickness=3)  # סגול
    cv2.polylines(image_color, [bottom_curve_pts], isClosed=False, color=(0, 255, 255), thickness=3)  # צהוב

    output_path = r"C:\Users\user\PycharmProjects\pythonProject10\output_marked_image.png"
    cv2.imwrite(output_path, image_color)
    print(f"תמונה נשמרה בהצלחה אל: {output_path}")

    return P4


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


# --- טעינת תמונה והרצה ---
image = cv2.imread(r"/testings & mid Results/Page_split_S2\right_page_0.png", cv2.IMREAD_GRAYSCALE)

_, binary0 = cv2.threshold(image, 170, 255, cv2.THRESH_BINARY_INV)
_, binary1 = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)

right_top_y = R_get_right_top_y(binary1)
right_top_x = R_get_right_top_x(binary1, right_top_y)

left_top_x = 0
left_top_y = R_get_top_left_y(binary0)
left_bottom_x = 0
left_bottom_y = R_get_bottom_left_y(binary0)

P1 = (left_top_x, left_top_y)
P2 = (right_top_x, right_top_y)
P3 = (left_bottom_x, left_bottom_y)

P4 = process(binary1, P1, P2, P3)
