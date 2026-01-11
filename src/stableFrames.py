import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from helpers.utils import show_img


images_motion_values_a_id = []

def show_pages_and_motion_diff(gray1, gray2, orig_frame, id):
    """ מחשב תנועה בין שני פריימים אפורים ושומר גם את התמונה המקורית """
    diff = cv2.absdiff(gray1, gray2)
    motion = np.sum(diff) / 1000000  # מדד לכמות התנועה
    images_motion_values_a_id.append((motion, id, orig_frame))

    return motion

def process_video_frame(frame, prev_gray, counter):
    """ מעבד פריים יחיד: מחשב תנועה ושומר את התמונה המקורית """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    show_pages_and_motion_diff(gray, prev_gray, frame, counter)
    return gray

def video_to_images(video_path, num_of_pages):
    num_of_pages = num_of_pages + round(num_of_pages*0.20)
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
        if counter % 5 == 0:  # כל חמש פריימים
            prev_gray = process_video_frame(frame, prev_gray, counter)
        counter += 1
    # plot_motion_graph()
    choose_images(num_of_pages)


def choose_images(quantity):
    """ בחירת תמונות לפי חלוקה שווה וחיפוש מינימום בכל חלק """
    images_motion_values_a_id.sort(key=lambda x: x[1])  # למיין לפי מספר פריים
    segment_size = len(images_motion_values_a_id) // quantity
    images_list_by_segments = []
    for i in range(quantity):
        start_idx = i * segment_size
        if i == quantity - 1:
            end_idx = len(images_motion_values_a_id)  # לוודא שלא יזרוק החוצה בסוף
        else:
            end_idx = (i + 1) * segment_size
        segment = images_motion_values_a_id[start_idx:end_idx]
        if len(segment) < 2:
            continue
        selected_points = find_two_min_points(segment)
        for point in selected_points:
            images_list_by_segments.append((point[2], point[1]))  # (תמונה מקורית, מספר פריים)
    display_images(images_list_by_segments)

def find_two_min_points(segment):
    """ מחפש שתי נקודות מינימום עם מרחק מינימלי של רבע מהקטע """
    segment_len = len(segment)
    min_required_distance = segment_len / 5
    sorted_segment = sorted(segment, key=lambda x: x[0])  # מיין לפי תנועה (מהקטן לגדול)
    for i in range(len(sorted_segment)):
        for j in range(i + 1, len(sorted_segment)):
            if abs(sorted_segment[i][1] - sorted_segment[j][1]) >= min_required_distance:
                return [sorted_segment[i], sorted_segment[j]]
    return sorted_segment[:2]

def display_images(images_list,
    output_dir=r"C:\Users\user\PycharmProjects\pythonProject10\testings & mid Results\Motion_detection_S1"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for idx, (image, frame_id) in enumerate(images_list):
        image_rotated = cv2.rotate(image, cv2.ROTATE_180)
        file_path = os.path.join(output_dir, f"image_{frame_id}.png")
        cv2.imwrite(file_path, image_rotated)


def plot_motion_graph():
    """ מצייר גרף של כמות התנועה לאורך הווידאו """
    motions, frame_ids, _ = zip(*images_motion_values_a_id)
    plt.figure(figsize=(10, 5))
    plt.plot(frame_ids, motions, marker='o', linestyle='-', color='b', label="Motion")
    plt.xlabel("Frame Number")
    plt.ylabel("Motion Level")
    plt.title("Motion Over Frames")
    plt.legend()
    plt.grid()
    plt.show()



