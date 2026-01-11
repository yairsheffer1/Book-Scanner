import os
import shutil
import re
from pathlib import Path

def extract_frame_number(filename):
    match = re.search(r'_(\d+)', filename)
    return int(match.group(1)) if match else None

def sync_folders_by_number(source_dir, target_dir):
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    # קבצים בתיקיות
    source_files = [f for f in os.listdir(source_dir) if f.endswith('.png')]
    target_files = [f for f in os.listdir(target_dir) if f.endswith('.png')]

    # מיפוי: מספר → קובץ
    source_numbers = {extract_frame_number(f): f for f in source_files if extract_frame_number(f) is not None}
    target_numbers = {extract_frame_number(f) for f in target_files if extract_frame_number(f) is not None}

    # חיפוש קבצים שמספרם לא מופיע בתיקיית היעד
    missing_numbers = set(source_numbers.keys()) - target_numbers

    print(f"🔍 נמצאו {len(missing_numbers)} קבצים חסרים. מעתיק...")

    for num in missing_numbers:
        src_filename = source_numbers[num]
        src_path = source_dir / src_filename
        dst_path = target_dir / src_filename
        shutil.copy(src_path, dst_path)
        print(f"✅ הועתק: {src_filename}")

    print("🎉 סנכרון הושלם.")


