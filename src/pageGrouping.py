import os
import shutil
import numpy as np
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pytesseract
import re
from sklearn.cluster import AgglomerativeClustering
import cv2
import Levenshtein

# הגדרת הנתיב ל־Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'

def get_image_text(image_path, lang='heb'):
    try:
        img = Image.open(image_path)
        return pytesseract.image_to_string(img, lang=lang).strip()
    except Exception as e:
        # print(f"Error reading image {image_path}: {e}")
        return ""

def compute_similarity(text1, text2):
    if not text1.strip() and not text2.strip():
        return 1.0
    if not text1.strip() or not text2.strip():
        return 0.0
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity
    except ValueError:
        return 0.0

def get_black_density(image_path, threshold=195):
    img = Image.open(image_path).convert("L")
    width, height = img.size
    top_half = img.crop((0, 0, width, height // 2))
    binary = top_half.point(lambda p: 0 if p < threshold else 255, mode='1')
    pixels = np.array(binary)
    total_pixels = pixels.size
    black_pixels = np.sum(pixels == 0)
    return (black_pixels / total_pixels) * 100

def extract_page_number(text):
    numbers = re.findall(r'\d+', text)
    if numbers:
        return int(numbers[0])
    return None

def compare_page_numbers(text1, text2):
    num1 = extract_page_number(text1)
    num2 = extract_page_number(text2)

    if num1 is None or num2 is None:
        return None  # לא מצאנו מספר עמוד בשני הצדדים

    if num1 == num2:
        return 1
    else:
        return 0

def extract_frame_number(image_path):
    filename = os.path.basename(image_path)
    match = re.search(r'_(\d+)\.png$', filename)
    if match:
        return int(match.group(1))
    else:
        return None

def compare_frame_numbers(image1_path, image2_path, total_frames, total_pages):
    frame1 = extract_frame_number(image1_path)
    frame2 = extract_frame_number(image2_path)
    if frame1 is None or frame2 is None:
        return None
    max_diff = total_frames / (total_pages * 2)
    diff = abs(frame1 - frame2)
    return 1 if diff <= max_diff else 0

def compare_pages(image1_path, image2_path, total_frames, total_pages, weights=None):
    if weights is None:
        weights = {
            'text_similarity': 0.2,
            'black_density': 0.2,
            'word_count': 0.2,
            'page_number': 0.5,
            'frame_number': 0.3
        }


    text1 = get_image_text(image1_path)
    text2 = get_image_text(image2_path)
    sim_text = compute_similarity(text1, text2)


    density1 = get_black_density(image1_path)
    density2 = get_black_density(image2_path)
    min_density = min(density1, density2)
    max_density = max(density1, density2)
    safe_min_density = max(min_density, 0.01)
    ratio = max_density / safe_min_density
    sim_density = 1 if ratio <= 3 else 0

    word_count1 = len(text1.split())
    word_count2 = len(text2.split())
    if min(word_count1, word_count2) == 0:
        sim_words = 0
    else:
        ratio_wc = max(word_count1, word_count2) / min(word_count1, word_count2)
        sim_words = 1 if ratio_wc <= 1.3 else 0

    sim_page = compare_page_numbers(text1, text2)
    if sim_page is None:
        page_weight = 0
        sim_page_score = 0
    else:
        page_weight = weights['page_number']
        sim_page_score = sim_page

    sim_frame = compare_frame_numbers(image1_path, image2_path, total_frames, total_pages)
    if sim_frame is None:
        final_score = (
            (weights['text_similarity'] * sim_text +
                weights['black_density'] * sim_density +
                weights['word_count'] * sim_words +
                page_weight * sim_page_score) * 1.2
        )
    else:
        frame_weight = weights['frame_number']
        sim_frame_score = sim_frame
        final_score = (
            weights['text_similarity'] * sim_text +
            weights['black_density'] * sim_density +
            weights['word_count'] * sim_words +
            page_weight * sim_page_score +
            frame_weight * sim_frame_score
        )

    # הדפסה לדיבאג (אפשר להסיר או להפעיל לפי צורך)
    # print(f"\nComparing {os.path.basename(image1_path)} and {os.path.basename(image2_path)}")
    # print(f"Similarity score: {final_score:.2f}")

    return final_score

def compute_ocr_accuracy(processed_path, reference_path):
    img1 = cv2.imread(processed_path)
    img2 = cv2.imread(reference_path)
    text1 = pytesseract.image_to_string(img1, lang='heb')
    text2 = pytesseract.image_to_string(img2, lang='heb')
    distance = Levenshtein.distance(text1, text2)
    max_len = max(len(text1), len(text2))
    if max_len == 0:
        return 1.0
    accuracy = 1 - distance / max_len
    return accuracy

def find_non_matching_groups(groups_L, groups_R, matching_pairs):
    matched_L = set(int(name.split()[-1]) - 1 for name, _ in matching_pairs)
    matched_R = set(int(name.split()[-1]) - 1 for _, name in matching_pairs)
    unmatched_L = {k: v for k, v in groups_L.items() if k not in matched_L}
    unmatched_R = {k: v for k, v in groups_R.items() if k not in matched_R}
    return unmatched_L, unmatched_R

def cluster_images_with_neighbors_only(image_paths, total_pages, total_frames,neighbors=2):
    n = len(image_paths)

    # אתחול מטריצת מרחק עם ערך מרחק גבוה ברירת מחדל
    distance_matrix = np.ones((n, n))

    # מחשבים דמיון רק לשכנים במרחק עד 'neighbors' לכל כיוון
    for i in range(n):
        distance_matrix[i, i] = 0
        start = max(0, i - neighbors)
        end = min(n, i + neighbors + 1)

        for j in range(start, end):
            if j <= i:
                continue  # כך תוודא שאתה משווה כל זוג רק פעם אחת

            score = compare_pages(image_paths[i], image_paths[j], total_frames=800, total_pages=total_pages)
            dist = 1 - score
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist  # ממשיך לשמור על הסימטריה

    # יצירת המודל, שים לב בגרסאות החדשות:
    clustering = AgglomerativeClustering(
        n_clusters=total_pages,
        metric='precomputed',  # affinity deprecated, metric נכון בגרסאות sklearn חדשות
        linkage='average'
    )

    labels = clustering.fit_predict(distance_matrix)

    # ארגון לפי תוויות
    groups = {}
    for label, path in zip(labels, image_paths):
        groups.setdefault(label, []).append(path)
    return groups, distance_matrix



def extract_frame_number_from_path(path):
    filename = os.path.basename(path)
    match = re.search(r'_(\d+)\.png$', filename)
    return int(match.group(1)) if match else -1

def find_fully_matching_groups(groups_L, groups_R):
    matches = []

    def group_to_frame_set(group):
        return set(extract_frame_number_from_path(p) for p in group)

    for i_l, group_L in enumerate(groups_L.values(), 1):
        frames_L = group_to_frame_set(group_L)
        for i_r, group_R in enumerate(groups_R.values(), 1):
            frames_R = group_to_frame_set(group_R)
            if frames_L == frames_R:
                matches.append((f"Left Group {i_l}", f"Right Group {i_r}"))

    return matches

def filter_group_to_most_text(groups, cropped_folder, lang='heb'):
    filtered_groups = {}

    # יצירת נתיב לתיקיית finals מחוץ ל־Cropped
    base_project_folder = os.path.dirname(cropped_folder)
    side_name = os.path.basename(cropped_folder).replace("Cropped_", "")
    side_name = os.path.basename(side_name).replace("_S3", "")  # לדוגמה: R_S3
    finals_folder = os.path.join(base_project_folder, f"finals_{side_name}_S4")
    os.makedirs(finals_folder, exist_ok=True)

    for i, (group_id, image_paths) in enumerate(groups.items(), 1):
        max_words = -1
        best_image = None

        for path in image_paths:
            text = get_image_text(path, lang=lang)
            word_count = len(text.split())

            if word_count > max_words:
                max_words = word_count
                best_image = path

        if best_image:
            filtered_groups[group_id] = [best_image]

            # העתקה לתיקיית finals החדשה
            filename = os.path.basename(best_image)
            dest_path = os.path.join(finals_folder, filename)
            shutil.copy2(best_image, dest_path)

    return filtered_groups

def extract_frame_number_from_filename(filename):
    match = re.search(r'_(\d+)\.png$', filename)
    return int(match.group(1)) if match else float('inf')  # כך קבצים בלי מספר יגיעו לסוף

def reorder_group_labels(groups_dict):
    def extract_frame_number(path):
        match = re.search(r'_(\d+)\.', os.path.basename(path))
        return int(match.group(1)) if match else float('inf')

    # מיון לפי הפריים הראשון בכל קבוצה
    sorted_items = sorted(groups_dict.items(), key=lambda item: extract_frame_number(item[1][0]) if item[1] else float('inf'))

    # מיפוי חדש: group_id -> קבוצת תמונות, עם group_id חדש לפי הסדר
    new_groups = {new_label: paths for new_label, (_, paths) in enumerate(sorted_items)}
    return new_groups

def plot_distance_matrix_and_dendrogram(distance_matrix, image_paths, title="Clustering Overview"):
    import matplotlib.pyplot as plt
    import seaborn as sns


    labels = [os.path.basename(p) for p in image_paths]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        distance_matrix,
        xticklabels=labels,
        yticklabels=labels,
        cmap="viridis",
        square=True,
        cbar_kws={"label": "Distance"}
    )
    plt.title(title, fontsize=14)
    plt.xlabel("Images")
    plt.ylabel("Images")
    plt.tight_layout()
    plt.show()

def cluster_images_with_ocr_only(image_paths, total_pages, neighbors=2):
    n = len(image_paths)
    distance_matrix = np.ones((n, n))

    for i in range(n):
        distance_matrix[i, i] = 0
        start = max(0, i - neighbors)
        end = min(n, i + neighbors + 1)

        for j in range(start, end):
            if j <= i:
                continue

            dist = 1 - compute_ocr_accuracy(image_paths[i], image_paths[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    clustering = AgglomerativeClustering(
        n_clusters=total_pages,
        metric='precomputed',
        linkage='average'
    )

    labels = clustering.fit_predict(distance_matrix)

    groups = {}
    for label, path in zip(labels, image_paths):
        groups.setdefault(label, []).append(path)

    return groups, distance_matrix


def run_full_processing(folder_R, folder_L, total_pages, total_frames, lang='heb'):
    # איסוף ומיון קבצים
    image_files_R = sorted(
        [f for f in os.listdir(folder_R) if f.lower().endswith('.png')],
        key=extract_frame_number_from_filename
    )
    image_paths_R = [os.path.join(folder_R, f) for f in image_files_R]

    image_files_L = sorted(
        [f for f in os.listdir(folder_L) if f.lower().endswith('.png')],
        key=extract_frame_number_from_filename
    )
    image_paths_L = [os.path.join(folder_L, f) for f in image_files_L]

    # קלאסטרינג ראשוני
    groups_L, dist_matrix_L = cluster_images_with_neighbors_only(image_paths_L, total_pages, total_frames, neighbors=2)
    groups_R, dist_matrix_R = cluster_images_with_neighbors_only(image_paths_R, total_pages, total_frames, neighbors=2)

    # סידור לפי כרונולוגיה
    groups_L = reorder_group_labels(groups_L)
    groups_R = reorder_group_labels(groups_R)

    # הדפסת קבוצות
    for label, paths in sorted(groups_L.items()):
        print(f"\n--- קבוצה {label + 1} (L) ---")
        for path in paths:
            print(path)
    for label, paths in sorted(groups_R.items()):
        print(f"\n--- קבוצה {label + 1} (R) ---")
        for path in paths:
            print(path)

    # שלב התאמה ראשון
    matching_pairs = find_fully_matching_groups(groups_L, groups_R)
    print("\n==== קבוצות תואמות לחלוטין ====")
    for left, right in matching_pairs:
        print(f"{left} == {right}")

    unmatched_L, unmatched_R = find_non_matching_groups(groups_L, groups_R, matching_pairs)

    # שלב OCR נוסף רק אם יש קבוצות לא מותאמות
    if unmatched_L and unmatched_R:
        image_paths_L2 = [img for group in unmatched_L.values() for img in group]
        image_paths_R2 = [img for group in unmatched_R.values() for img in group]

        if image_paths_L2 and image_paths_R2 and len(unmatched_L) > 0 and len(unmatched_R) > 0:
            groups_L2, _ = cluster_images_with_ocr_only(image_paths_L2, total_pages=len(unmatched_L), neighbors=2)
            groups_R2, _ = cluster_images_with_ocr_only(image_paths_R2, total_pages=len(unmatched_R), neighbors=2)

            groups_L2 = reorder_group_labels(groups_L2)
            groups_R2 = reorder_group_labels(groups_R2)

            new_matches = find_fully_matching_groups(groups_L2, groups_R2)

            if new_matches:
                print("\n==== התאמות חדשות לפי OCR clustering ====")
                for left, right in new_matches:
                    print(f"{left} == {right}")
                matching_pairs += new_matches

            # מחיקת קבוצות לא תואמות ישנות
            for k in unmatched_L:
                del groups_L[k]
            for k in unmatched_R:
                del groups_R[k]

            # השלמת קבוצות חסרות
            groups_L = fill_missing_groups(groups_L, groups_L2, matching_pairs, side='Left')
            groups_R = fill_missing_groups(groups_R, groups_R2, matching_pairs, side='Right')

    # סינון לפי כמות טקסט
    filtered_L = filter_group_to_most_text(groups_L, folder_L, lang=lang)
    filtered_R = filter_group_to_most_text(groups_R, folder_R, lang=lang)

    return filtered_L, filtered_R, matching_pairs, dist_matrix_L, dist_matrix_R, image_paths_L, image_paths_R


def fill_missing_groups(original_groups, new_groups, matching_pairs, side):
    filled_groups = original_groups.copy()
    existing_ids = set(filled_groups.keys())

    def frame_number(path):
        match = re.search(r'_(\d+)\.png$', os.path.basename(path))
        return int(match.group(1)) if match else float('inf')

    sorted_new_groups = sorted(new_groups.items(), key=lambda item: frame_number(item[1][0]))
    next_id = 0
    for _, paths in sorted_new_groups:
        while next_id in existing_ids:
            next_id += 1
        filled_groups[next_id] = paths
        existing_ids.add(next_id)

    return filled_groups




