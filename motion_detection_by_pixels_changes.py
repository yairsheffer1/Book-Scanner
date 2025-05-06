import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

images_motion_values_a_id = []


def show_pages_and_motion_diff(image1, image2, id):
    """מציג את שתי התמונות אחת ליד השנייה ואת כמות ההבדלים ביניהן"""
    diff = cv2.absdiff(image1, image2)
    motion = np.sum(diff) / 1000000  # מדד לכמות התנועה
    images_motion_values_a_id.append((motion, id, image1))  # שמירה עבור הגרף

    resized_frame1 = cv2.resize(image1, (600, 800))
    resized_frame2 = cv2.resize(image2, (600, 800))
    combined = np.hstack((resized_frame1, resized_frame2))

    text = f"Motion: {motion},{id}"
    cv2.putText(combined, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return motion


def process_video_frame(frame, prev_gray, counter):
    """מעבד פריים יחיד ומחשב תנועה"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    show_pages_and_motion_diff(gray, prev_gray, counter)
    return gray


def video_to_images(video_path, num_of_pages):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()

    if not ret:
        print("Error: Couldn't read video file.")
        return

    prev_gray = cv2.cvtColor(prev_frame.copy(), cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if counter % 5 == 0:
            prev_gray = process_video_frame(frame, prev_gray, counter)
        counter += 1

    cap.release()
    cv2.destroyAllWindows()
    plot_motion_graph()
    choose_images(num_of_pages, counter)


def find_max_points(quantity, video_len, min_dist):
    """מוצא את נקודות המקסימום עם מרחק מינימלי ביניהן"""
    max_points = [-10, video_len + 10]
    # max_points = []
    for i in images_motion_values_a_id[::-1]:
        if all(abs(i[1] - l) >= min_dist * 0.6 for l in max_points):
            max_points.append(i[1])
        if len(max_points) == quantity + quantity//3:
            break
    max_points.sort()
    return max_points


def find_min_points_between_max(left, right):
    """מוצא שתי נקודות מינימום בטווח בין נקודות מקסימום סמוכות"""
    candidates = [j for j in images_motion_values_a_id if left < j[1] < right]
    candidates.sort(key=lambda x: x[0])

    min_points = []
    for p1 in candidates:
        if not min_points:
            min_points.append(p1)
        elif len(min_points) == 1 and abs(p1[1] - min_points[0][1]) >= 20:
            min_points.append(p1)
            break

    return min_points


def choose_images(quantity, video_len):
    """בחירת תמונות לפי מדדי מינימום ומקסימום"""
    min_dist = video_len / (quantity * 2)
    images_motion_values_a_id.sort(key=lambda x: x[0])
    images_list_by_max = []

    max_points = find_max_points(quantity, video_len, min_dist)

    for i in range(len(max_points) - 1):
        left, right = max_points[i], max_points[i + 1]
        min_points = find_min_points_between_max(left, right)
        for point in min_points:
            images_list_by_max.append([point[2], point[1]])
    # plot_motion_graph()
    display_images(images_list_by_max)


def display_images(images_list, output_dir="selected_images"):
    """מציג ושומר את התמונות שנבחרו"""
    # צור את התיקיה אם היא לא קיימת
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, (image, frame_id) in enumerate(images_list):
        print(f"Saving frame {frame_id}...")

        # שמירת התמונה בשם ייחודי
        file_path = os.path.join(output_dir, f"selected_image_{frame_id}.png")
        cv2.imwrite(file_path, image)

        # הצגת התמונה
        # resized_frame = cv2.resize(image, (600, 800))
        # cv2.imshow("Selected Image", resized_frame)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    print(f"All selected images have been saved in the '{output_dir}' directory.")


def plot_motion_graph():
    """מצייר גרף של כמות התנועה לאורך הווידאו"""
    motions, frame_ids, _ = zip(*images_motion_values_a_id)
    plt.figure(figsize=(10, 5))
    plt.plot(frame_ids, motions, marker='o', linestyle='-', color='b', label="Motion")
    plt.xlabel("Frame Number")
    plt.ylabel("Motion Level")
    plt.title("Motion Over Frames")
    plt.legend()
    plt.grid()
    plt.show()



# video_to_images(r"C:\Users\user\Desktop\project\0.mp4", 3)
