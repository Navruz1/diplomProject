import threading
import tkinter as tk
from tkinter import filedialog

from video.video_capture import VideoSource
from video.async_pipeline import AsyncVideoPipeline
from model.detector import PersonDetector
from utils.image_utils import cv_to_tk


class YOLOVideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO – Детекция людей")

        bg = "#ffffff"      # background
        fg = "#2b2b2b"      # font-color
        abb = "#dfdfdf"     # active-button background
        self.root.configure(bg=bg)

        self.root.geometry("900x550")    # ширина x высота
        self.root.minsize(600, 400)      # минимальный размер
        self.root.resizable(True, True)  # Менять размер окна можно

        controls = tk.Frame(root, bg=bg)
        controls.pack(pady=10)

        tk.Button(controls, text="Открыть видео", bg=bg, fg=fg, activebackground=abb, command=self.open_video).pack(side=tk.LEFT, padx=5)
        tk.Button(controls, text="Старт", bg=bg, fg=fg, activebackground=abb, command=self.start).pack(side=tk.LEFT, padx=5)
        tk.Button(controls, text="Стоп", bg=bg, fg=fg, activebackground=abb, command=self.stop).pack(side=tk.LEFT, padx=5)

        self.video = VideoSource()
        self.detector = PersonDetector()

        self.running = False
        self.pipeline = None
        self.update_ui()

        self.video_label = tk.Label(root, bg=bg)
        self.video_label.pack()


    def open_video(self):
        path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mkv")]
        )
        if path:
            self.video.open(path)
            #
            # w, h = self.video.get_size()
            # if w and h:
            #     self.root.geometry(f"{w}x{h + 80}") # +80 - место под кнопки

        # загрузка первого кадра
        ret, frame = self.video.read_first_frame()
        if ret:
            w = self.root.winfo_width()
            h = self.root.winfo_height() - 80

            imgtk = cv_to_tk(frame, (w, h))

            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

    def start(self):
        if self.video.cap and not self.running:
            self.running = True
            self.pipeline = AsyncVideoPipeline(self.video, self.detector)
            self.pipeline.start()

    def stop(self):
        self.running = False
        if self.pipeline:
            self.pipeline.stop()

    def loop(self):
        while self.running:
            ret, frame = self.video.read()
            if not ret:
                break

            frame = self.detector.draw(frame)
            window_w = self.root.winfo_width()
            window_h = self.root.winfo_height() - 80

            imgtk = cv_to_tk(frame, (window_w, window_h))

            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.running = False

    def update_ui(self):
        if self.pipeline and not self.pipeline.output_queue.empty():
            frame, result = self.pipeline.output_queue.get()
            imgtk = cv_to_tk(frame, (
                self.root.winfo_width(),
                self.root.winfo_height() - 80
            ))

            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            if result.boxes.id is not None:
                ids = result.boxes.id.cpu().tolist()
                print("IDs:", ids)

        self.root.after(15, self.update_ui)