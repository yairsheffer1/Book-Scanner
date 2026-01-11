import cv2
import matplotlib.pyplot as plt

def show_img(img, title=None):
    if title:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.title(title)
        plt.axis('off')
        plt.show()

def open_img(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found! Check the file path.")
    return image
