import cv2
import matplotlib.pyplot as plt

video_path = r"C:\Users\user\Desktop\project\7_HaimSheli.M4V"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Original FPS: {fps}")

delay = 1 / (fps * 1.5)  # seconds per frame

plt.ion()  # interactive mode
fig, ax = plt.subplots()
image_disp = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if image_disp is None:
        image_disp = ax.imshow(frame_rgb)
        plt.axis('off')
    else:
        image_disp.set_data(frame_rgb)

    plt.pause(delay)

cap.release()
plt.ioff()
plt.close()
