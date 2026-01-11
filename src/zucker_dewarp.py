import subprocess
from pathlib import Path


def run0(input_path, output_path):
    input_folder = Path(input_path)
    output_folder = Path(output_path)

    page_dewarp_cmd = "page-dewarp"

    # סיומות מותאמות לקבצי תמונה
    valid_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    for file_path in input_folder.iterdir():
        if file_path.suffix.lower() in valid_extensions:
            print(f"🔄 מריץ על: {file_path.name}")
            command = [
                page_dewarp_cmd,
                "-x", "0",  # בלי שוליים לרוחב
                "-y", "0",
                "-s", "1",
                "-nb", "1",
                str(file_path.resolve())
            ]
            subprocess.run(command, cwd=output_folder)

def run1(input_path, output_path):
    input_folder = Path(input_path)
    output_folder = Path(output_path)

    page_dewarp_cmd = "page-dewarp"

    # סיומות מותאמות לקבצי תמונה
    valid_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    for file_path in input_folder.iterdir():
        if file_path.suffix.lower() in valid_extensions:
            print(f"🔄 מריץ על: {file_path.name}")
            command = [
                page_dewarp_cmd,
                "-x", "0",  # בלי שוליים לרוחב
                "-y", "0",
                "-s", "1",
                str(file_path.resolve())
            ]
            subprocess.run(command, cwd=output_folder)
