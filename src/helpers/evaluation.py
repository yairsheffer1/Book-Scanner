import cv2
from skimage.metrics import structural_similarity as ssim
import pytesseract
import os
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'

def compute_ssim_score(processed_path, reference_path):
    img1 = cv2.imread(processed_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        print(f"Error loading images for SSIM: {processed_path}, {reference_path}")
        return 0.0
    img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))
    score, _ = ssim(img1, img2, full=True)
    return score

def compute_ocr_accuracy(processed_path, reference_path):
    img1 = cv2.imread(processed_path)
    img2 = cv2.imread(reference_path)
    text1 = pytesseract.image_to_string(img1, lang='heb')
    text2 = pytesseract.image_to_string(img2, lang='heb')
    print(text1,text2)
    # Levenshtein Distance
    import Levenshtein
    distance = Levenshtein.distance(text1, text2)
    max_len = max(len(text1), len(text2))
    if max_len == 0:
        return 1.0
    accuracy = 1 - distance / max_len
    return accuracy


s = compute_ssim_score(
    r"/testings & mid Results/filesOfPdf\cropped_L_0_sharpened.png",
    r"C:\Users\user\Desktop\project\Yadid_Nefesh.png"
)
o = compute_ocr_accuracy(
    r"/testings & mid Results/finals1_Dewarped_R_S4\cropped_R_405_thresh.png",
    r"/testings & mid Results/finals_R_S4\cropped_R_405.png"
)

print(f"SSIM={s:.3f}, OCR Accuracy={o:.3f}, pHash Similarity={p:.3f}")
