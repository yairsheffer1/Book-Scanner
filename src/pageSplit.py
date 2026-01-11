
import glob
import cv2
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from helpers.utils import open_img,show_img



def show_projection_graph(left,  right,projection,num):
    right = right
    middle = (left + right) // 2
    plt.figure(figsize=(10, 5))
    plt.title(f"{num}_Projection Profile")
    plt.plot(projection, label="Projection")
    plt.axvline(middle, color='r', linestyle='--', label=f'Selected Middle: {middle}')
    plt.axvline(left, color='g', linestyle='--', label=f'Selected left: {left}')
    plt.axvline(right, color='g', linestyle='--', label=f'Selected Middle: {right}')
    plt.legend()
    plt.show()


def find_page_edge(start,end,step,maxx,projection):
    find = start
    for ind in range(start, end, step):
        i = int(projection[ind])
        if i > maxx:
            maxx = i
        if abs(i - maxx) <= maxx * 0.2:
            find = ind
            maxx = i
        else:
            break
    return find


def page_split(image_path, num=""):
    image = open_img(image_path)
    image_colored = cv2.imread(image_path, cv2.IMREAD_COLOR)
    _, binary = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)
    # show_img(binary)

    projection = np.sum(binary, axis=0)
    length = len(projection)
    step = max(1, int(length * 0.01))
    right = find_page_edge(length - 2,length // 2, -step, int(projection[length - 1]), projection)
    left = find_page_edge(1,length // 2, step, int(projection[0]), projection)
    middle = (left + right) // 2
    # show_projection_graph(left, right ,projection, middle)
    save_and_split(image_colored,middle, num)


def save_and_split(img,middle, num):
    left_page = img[:, :middle]
    right_page = img[:, middle:]
    output_dir = r"C:\Users\user\PycharmProjects\pythonProject10\testings & mid Results\Page_split_S2"
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, f"left_page_{num}.png"), left_page)
    cv2.imwrite(os.path.join(output_dir, f"right_page_{num}.png"), right_page)


def folder_split(folder_path):
    if not os.path.exists(folder_path):
        raise ValueError("Folder not found! Check the path.")
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for file_name in sorted(image_files):
        match = re.search(r'(\d+)', file_name)
        if match:
            image_num = match.group(1)
        else:
            continue
        full_path = os.path.join(folder_path, file_name)
        page_split(full_path, num=image_num)


def delete_outliers_pages(folder, side='left', tolerance=0.10):
    pattern = os.path.join(folder, f"{side}_page_*.png")
    page_files = glob.glob(pattern)
    lengths = []
    for f in page_files:
        img = cv2.imread(f)
        if img is not None:
            lengths.append(img.shape[1])

    avg_length = sum(lengths) / len(lengths)
    min_length = avg_length * (1 - tolerance)
    max_length = avg_length * (1 + tolerance)

    delete_img(page_files, ( min_length, max_length))


def delete_img(pages,borders):
    min_length, max_length = borders
    for f in pages:
        img = cv2.imread(f)
        if img is not None:
            width = img.shape[1]
            if width < min_length or width > max_length:
                print(f"Deleting file: {os.path.basename(f)} with width: {width}")
                # utils.show_img(img,f"Deleting: {os.path.basename(f)} (width={width})")
                os.remove(f)



if __name__ == "__main__":
    # page_split(r"C:\Users\user\PycharmProjects\pythonProject10\Motion_detection_S1")
    folder_split(r"C:\Users\user\PycharmProjects\pythonProject10\testings & mid Results\Motion_detection_S1")