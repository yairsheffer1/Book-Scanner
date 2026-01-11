import os
import re
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import mask2
import mask
import pytesseract
from PyPDF2 import PdfMerger
import tempfile
import subprocess


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'

# ---------- Utility Functions ----------

def extract_frame_number(path):
    match = re.search(r'_(\d+)', os.path.basename(path))
    return int(match.group(1)) if match else float('inf')

def extract_group_index(filename):
    match = re.search(r'_(\d+)', filename)
    return int(match.group(1)) if match else float('inf')


def apply_filters(img_cv):
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    filters = {
        'original': img_cv.copy(),
        'sharpened': cv2.filter2D(img_cv, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])),
    }

    otsu_thresh, _ = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adjusted_thresh = max(0, otsu_thresh + 20)
    _, binary_darker = cv2.threshold(img_gray, adjusted_thresh, 255, cv2.THRESH_BINARY)
    binary_color = cv2.cvtColor(binary_darker, cv2.COLOR_GRAY2BGR)
    filters['binary'] = binary_color

    binary_matrix = (binary_darker == 0).astype(np.uint8)
    cleaned_mask = mask2.flood_fill_from_corners(binary_matrix.copy())
    mask_to_white = (cleaned_mask == 0)

    for key in filters:
        if key != 'original':
            img = filters[key]
            img[mask_to_white] = [255, 255, 255]

    return filters


def create_multiple_pdfs_with_filters(output_base_path, filtered_L, filtered_R, matching_pairs):
    def merge_and_order_pages(filtered_L, filtered_R, matching_pairs):
        left_to_right = {}
        right_to_left = {}
        for left_label, right_label in matching_pairs:
            l_idx = extract_group_index(left_label)
            r_idx = extract_group_index(right_label)
            left_to_right[l_idx] = r_idx
            right_to_left[r_idx] = l_idx

        all_indices = sorted(set(filtered_L.keys()) | set(filtered_R.keys()))
        added = set()
        ordered_pages = []

        for idx in all_indices:
            if idx in right_to_left:
                left_idx = right_to_left[idx]
                right_idx = idx
            elif idx in left_to_right:
                left_idx = idx
                right_idx = left_to_right[idx]
            else:
                left_idx = idx if idx in filtered_L else None
                right_idx = idx if idx in filtered_R else None

            if left_idx in filtered_L and right_idx in filtered_R and (left_idx, right_idx) in matching_pairs:
                for p in sorted(filtered_R[right_idx], key=extract_frame_number):
                    if p not in added:
                        ordered_pages.append(p)
                        added.add(p)
                for p in sorted(filtered_L[left_idx], key=extract_frame_number):
                    if p not in added:
                        ordered_pages.append(p)
                        added.add(p)
            else:
                if right_idx is not None and right_idx in filtered_R:
                    for p in sorted(filtered_R[right_idx], key=extract_frame_number):
                        if p not in added:
                            ordered_pages.append(p)
                            added.add(p)
                if left_idx is not None and left_idx in filtered_L:
                    for p in sorted(filtered_L[left_idx], key=extract_frame_number):
                        if p not in added:
                            ordered_pages.append(p)
                            added.add(p)
        return ordered_pages

    ordered_pages = merge_and_order_pages(filtered_L, filtered_R, matching_pairs)
    pil_images_by_filter = {key: [] for key in ['original', 'sharpened', 'binary']}
    output_dir = "../testings & mid Results/filesOfPdf"

    for img_path in ordered_pages:
        img_cv = cv2.imread(img_path)
        if img_cv is None:
            print(f"Failed to load image: {img_path}")
            continue

        filtered_versions = apply_filters(img_cv)

        for key in filtered_versions:
            if key == 'sharpened':
                save_filename = f"{Path(img_path).stem}_sharpened.png"
                if "_R_" in save_filename:
                    side_dir = os.path.join(output_dir, "R")
                elif "_L_" in save_filename:
                    side_dir = os.path.join(output_dir, "L")
                else:
                    side_dir = output_dir

                os.makedirs(side_dir, exist_ok=True)
                save_path = os.path.join(side_dir, save_filename)
                cv2.imwrite(save_path, filtered_versions['sharpened'])
                print(f"✅ Saved: {save_path}")

        for key, filtered_img in filtered_versions.items():
            img_rgb = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            pil_images_by_filter[key].append(pil_img)

    for key, pil_images in pil_images_by_filter.items():
        if not pil_images:
            print(f"No images found for filter: {key}")
            continue
        output_pdf_path = f"{output_base_path}_{key}.pdf"
        pil_images[0].save(output_pdf_path, save_all=True, append_images=pil_images[1:])
        print(f"Created PDF for filter '{key}': {output_pdf_path}")



def create_interleaved_pdf_by_number(right_folder, left_folder, output_pdf_path):
    def collect_and_sort(folder):
        return sorted(
            [f for f in os.listdir(folder) if f.endswith('.png')],
            key=extract_frame_number
        )

    right_images = collect_and_sort(right_folder)
    left_images = collect_and_sort(left_folder)

    combined = []
    max_len = max(len(right_images), len(left_images))
    for i in range(max_len):
        if i < len(right_images):
            combined.append(os.path.join(right_folder, right_images[i]))
        if i < len(left_images):
            combined.append(os.path.join(left_folder, left_images[i]))

    pil_images = []
    for img_path in combined:
        try:
            img = Image.open(img_path).convert("RGB")
            pil_images.append(img)
        except Exception as e:
            print(f"⚠️ Failed to open {img_path}: {e}")

    if pil_images:
        pil_images[0].save(output_pdf_path, save_all=True, append_images=pil_images[1:])
        print(f"📄 PDF נוצר בהצלחה: {output_pdf_path}")
    else:
        print("⚠️ לא נמצאו תמונות ליצירת PDF.")


def collect_images_in_pairs(folderR, folderL):
    valid_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}

    def sorted_images(folder):
        return sorted([
            str(p) for p in Path(folder).iterdir()
            if p.suffix.lower() in valid_extensions
        ])

    right_images = sorted_images(folderR)
    left_images = sorted_images(folderL)

    max_len = max(len(right_images), len(left_images))
    ordered = []

    for i in range(max_len):
        if i < len(right_images):
            ordered.append(right_images[i])
        if i < len(left_images):
            ordered.append(left_images[i])

    return ordered

def create_ocr_pdf_group(source_name, folderR, folderL):
    image_paths = collect_images_in_pairs(folderR, folderL)
    filters_to_generate = ['binary', 'sharpened']

    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_paths = {flt: [] for flt in filters_to_generate}

        for idx, img_path in enumerate(image_paths):
            img_cv = cv2.imread(img_path)
            if img_cv is None:
                print(f"❌ לא נטענה תמונה: {img_path}")
                continue

            filtered_versions = apply_filters(img_cv)

            for flt in filters_to_generate:
                # filtered = shift_image_to_center_text(filtered_versions[flt])
                temp_img_path = os.path.join(temp_dir, f"{flt}_{idx}.png")
                cv2.imwrite(temp_img_path, flt)

                temp_pdf_path = os.path.join(temp_dir, f"{flt}_{idx}.pdf")

                try:
                    subprocess.run([
                        pytesseract.pytesseract.tesseract_cmd,
                        temp_img_path,
                        temp_pdf_path[:-4],
                        "-l", "heb",
                        "--psm", "6",
                        "pdf"
                    ], check=True)
                    pdf_paths[flt].append(temp_pdf_path)
                    print(f"📄 OCR בוצע: {flt}_{idx}")
                except subprocess.CalledProcessError as e:
                    print(f"⚠️ שגיאה ב-OCR לתמונה: {img_path} — {e}")

        for flt, paths in pdf_paths.items():
            if not paths:
                print(f"⚠️ אין PDFים עבור: {source_name}_{flt}")
                continue

            output_pdf_path = f"{source_name}_{flt}.pdf"
            merger = PdfMerger()
            for p in paths:
                merger.append(p)
            merger.write(output_pdf_path)
            merger.close()
            print(f"✅ PDF סופי נוצר: {output_pdf_path}")


def apply_masks_to_folder(input_folder, output_base_folder):
    valid_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    os.makedirs(output_base_folder, exist_ok=True)

    for img_file in os.listdir(input_folder):
        if Path(img_file).suffix.lower() not in valid_extensions:
            continue

        img_path = os.path.join(input_folder, img_file)
        img_cv = cv2.imread(img_path)
        if img_cv is None:
            print(f"❌ לא נטענה תמונה: {img_path}")
            continue

        img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        _, binary_darker = cv2.threshold(img_gray, 160, 255, cv2.THRESH_BINARY)

        binary_matrix = (binary_darker == 0).astype(np.uint8)
        cleaned_mask2 = mask.flood_fill_from_corners(binary_matrix.copy())

        mask2_applied = img_cv.copy()
        mask2_applied[cleaned_mask2 == 1] = [255, 255, 255]

        save_dir = output_base_folder
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{Path(img_file).stem}.png")
        cv2.imwrite(save_path, mask2_applied)
        print(f"✅ נשמר: {save_path}")

# ✨ הפעלת הקוד — שנה את הנתיבים בהתאם
# if __name__ == "__main__":
#     create_all_ocr_pdfs()
