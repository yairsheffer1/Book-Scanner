import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import subprocess
from main import run_full_processing  # תוודא שזה הנתיב למודול שלך


def open_results_folder(path):
    try:
        if os.name == 'nt':  # Windows
            os.startfile(path)
        elif os.name == 'posix':  # macOS, Linux
            subprocess.call(['open', path])
        else:
            messagebox.showinfo("Info", f"Please open manually:\n{path}")
    except Exception as e:
        messagebox.showerror("Error", f"Cannot open folder:\n{e}")


class BookScannerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Book Scanner")
        self.root.geometry("800x600")
        self.frames = {}

        self.create_start_screen()

    def clear_screen(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def create_start_screen(self):
        self.clear_screen()
        frame = tk.Frame(self.root)
        frame.pack(expand=True)

        title = tk.Label(frame, text="📖 Book Scanner", font=("Helvetica", 32, "bold"))
        title.pack(pady=40)

        start_button = tk.Button(
            frame,
            text="Let's Start",
            font=("Helvetica", 18),
            command=self.create_instruction_screen
        )
        start_button.pack(pady=20)

    def create_instruction_screen(self):
        self.clear_screen()
        frame = tk.Frame(self.root)
        frame.pack(expand=True, fill="both")

        instructions = (
            "Instructions:\n\n"
            "- Make sure the video is recorded against a dark background.\n"
            "- Ensure the video clearly shows flipping through all pages.\n"
            "- After each page turn, hold the page from the lower corners for at least half a second.\n"
            "- Upload the video file to the system.\n"
            "- Enter the total number of pages.\n"
            "- Click 'Start Processing' to begin.\n"
        )

        label = tk.Label(frame, text=instructions, font=("Helvetica", 14), justify="left")
        label.pack(pady=10)

        try:
            img = Image.open(r"C:\Users\user\Desktop\project\ChatGPT Image Jun 29, 2025, 11_58_41 PM.png")
            img = img.resize((300, 200), Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)
            image_label = tk.Label(frame, image=img_tk)
            image_label.image = img_tk
            image_label.pack(pady=10)
        except FileNotFoundError:
            tk.Label(frame, text="(Example image not found)", fg="gray").pack()

        continue_button = tk.Button(
            frame,
            text="Continue",
            font=("Helvetica", 16),
            command=self.create_main_input_screen
        )
        continue_button.pack(pady=20)

    def create_main_input_screen(self):
        self.clear_screen()

        frame = tk.Frame(self.root)
        frame.pack(expand=True)

        tk.Label(frame, text="Select Video File:", font=("Helvetica", 12)).grid(row=0, column=0, padx=10, pady=10, sticky="e")
        self.entry_video = tk.Entry(frame, width=50)
        self.entry_video.grid(row=0, column=1, padx=10)
        tk.Button(frame, text="Browse", command=self.browse_file).grid(row=0, column=2, padx=10)

        tk.Label(frame, text="Number of Pages:", font=("Helvetica", 12)).grid(row=1, column=0, padx=10, pady=10, sticky="e")
        self.entry_pages = tk.Entry(frame)
        self.entry_pages.insert(0, "9")
        self.entry_pages.grid(row=1, column=1, padx=10)

        tk.Button(
            frame,
            text="Start Processing",
            font=("Helvetica", 14),
            command=self.run_processing
        ).grid(row=2, column=0, columnspan=3, pady=30)

    def browse_file(self):
        file_selected = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.M4V *.mov")])
        if file_selected:
            self.entry_video.delete(0, tk.END)
            self.entry_video.insert(0, file_selected)

    def show_loading_popup(self):
        self.loading_window = tk.Toplevel(self.root)
        self.loading_window.title("Processing")
        self.loading_window.geometry("300x120")
        self.loading_window.resizable(False, False)
        tk.Label(
            self.loading_window,
            text="Processing...\nThis may take a few minutes.\nPlease wait.",
            font=("Helvetica", 12),
            justify="center"
        ).pack(expand=True, pady=20)

    def run_processing(self):
        video_path = self.entry_video.get()
        try:
            num_pages = int(self.entry_pages.get())
        except ValueError:
            messagebox.showerror("Error", "The number of pages must be an integer.")
            return

        if not os.path.isfile(video_path):
            messagebox.showerror("Error", "Please select a valid video file.")
            return

        self.show_loading_popup()
        self.root.update()

        try:
            run_full_processing(video_path, num_pages)
            self.loading_window.destroy()

            result = messagebox.askyesno("Success", "Processing completed successfully!\n\nDo you want to open the results folder?")
            if result:
                open_results_folder(r"C:\Users\user\PycharmProjects\pythonProject10\results")  # תעדכן את הנתיב שלך

        except Exception as e:
            self.loading_window.destroy()
            messagebox.showerror("Error", f"An error occurred during processing:\n{e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = BookScannerApp(root)
    root.mainloop()
