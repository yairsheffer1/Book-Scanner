import os
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# הנתיב לתיקייה המקורית
folder_path = r"../../testings & mid Results/Motion_detection_S1"

# תיקיות הפלט
output_folder = r"C:\Users\user\PycharmProjects\pythonProject10\Page_split_S2_processed"
mask_folder = r"C:\Users\user\PycharmProjects\pythonProject10\Page_split_S2_masks"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(mask_folder, exist_ok=True)

# טען את המודל
model = YOLO("../../out resources/yolov8n-seg.pt")

for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        image_path = os.path.join(folder_path, filename)
        print(f"Processing {image_path}...")

        # טען תמונה
        original = Image.open(image_path).convert("RGB")
        np_image = np.array(original)
        orig_h, orig_w = np_image.shape[:2]

        # תמונה חדשה
        output_image = np_image.copy()
        final_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)  # לצבירת המסכה

        # הרצת YOLO
        results = model(image_path)

        for r in results:
            if r.masks is not None:
                masks = r.masks.data.cpu().numpy()
                class_ids = r.boxes.cls.cpu().numpy().astype(int)
                names = model.names

                for i, mask in enumerate(masks):
                    class_name = names[class_ids[i]]
                    if "person" in class_name.lower():
                        mask_resized = cv2.resize(mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                        mask_binary = (mask_resized > 0.1).astype(np.uint8) * 255
                        ys, xs = np.where(mask_binary == 255)
                        if len(ys) > 0:
                            highest_person_pixel = ys.min()
                            if highest_person_pixel < orig_h // 2:
                                print(f"הפיקסל הכי גבוה שבו זוהה אדם: {highest_person_pixel}")
                        else:
                            print("לא זוהה אף פיקסל כאדם בתמונה הזו.")

                        final_mask = cv2.bitwise_or(final_mask, mask_binary)

                        # צבע את היד בלבן מוחלט
                        output_image[mask_binary == 255] = [255, 255, 255]

        # המרת התמונה לגווני אפור
        gray = cv2.cvtColor(output_image, cv2.COLOR_RGB2GRAY)

        # שמירת התמונה המעובדת
        processed_path = os.path.join(output_folder, filename)
        cv2.imwrite(processed_path, gray)

        # שמירת המסכה (שחור/לבן)
        mask_path = os.path.join(mask_folder, f"mask_{filename}")
        cv2.imwrite(mask_path, final_mask)

print("Done!")
