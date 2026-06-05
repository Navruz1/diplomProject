import os
import tkinter as tk
from tkinter import filedialog

import cv2

from model.detector import PersonDetector
from model.payment_analyzer import PaymentAnalyzer
from ui.layout.player import VideoPlayer
from ui.layout.sidebar import Sidebar
from ui.widgets.tooltip import attach_tooltip
from utils.image_utils import cv_to_tk
from video.async_pipeline import AsyncVideoPipeline
from video.video_capture import VideoSource


class AppController:
    def __init__(self, root):
        self.root = root

        self.video = VideoSource()
        self.detector = PersonDetector()
        self.pipeline = None
        self.running = False

        self.first_frame = None
        self.terminal_zone = None
        self.zone_start = None
        self.display_meta = None

        self.video_total_frames = 0
        self.slider_dragging = False
        self.was_running_before_seek = False
        self.suppress_slider_command = False

        self.sidebar = Sidebar(root)
        self.sidebar.pack(side="left", fill="y")

        self.player = VideoPlayer(root)
        self.player.pack(side="right", fill="both", expand=True)

        self.sidebar.add_button.config(command=self.open_video)
        attach_tooltip(self.sidebar.add_button, "Открыть видео")

        self.sidebar.video_list.bind(
            "<Double-Button-1>", self.on_video_select
        )

        self.player.start_btn.config(command=self.start)
        self.player.stop_btn.config(command=self.stop)
        self.player.video_label.bind("<ButtonPress-1>", self.on_zone_press)
        self.player.video_label.bind("<B1-Motion>", self.on_zone_drag)
        self.player.video_label.bind("<ButtonRelease-1>", self.on_zone_release)
        self.player.progress_scale.config(command=self.on_progress_change)
        self.player.progress_scale.bind("<ButtonPress-1>", self.on_progress_press)
        self.player.progress_scale.bind("<ButtonRelease-1>", self.on_progress_release)
        self.root.bind_all("<space>", self.toggle_playback)

        self.current_folder = None
        self.update_ui()

    def open_video(self):
        path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mkv")]
        )
        if not path:
            return

        self.stop()
        self.video.open(path)
        self.load_folder(path)
        self.load_first_frame()

    def load_folder(self, path):
        self.sidebar.video_list.delete(0, tk.END)
        self.current_folder = os.path.dirname(path)
        self.sidebar.set_folder_path(self.current_folder)

        for file in sorted(os.listdir(self.current_folder)):
            if file.lower().endswith((".mp4", ".avi", ".mkv")):
                self.sidebar.video_list.insert(tk.END, file)

    def on_video_select(self, event):
        if not self.sidebar.video_list.curselection():
            return

        filename = self.sidebar.video_list.get(
            self.sidebar.video_list.curselection()
        )
        path = os.path.join(self.current_folder, filename)

        self.stop()
        self.video.open(path)
        self.load_first_frame()

    def load_first_frame(self):
        self.terminal_zone = None
        self.zone_start = None
        self.video_total_frames = self.video.get_frame_count()
        self.configure_progress(self.video_total_frames)

        ret, frame = self.video.read_first_frame()
        if not ret:
            self.first_frame = None
            self.player.status_label.config(text="Не удалось прочитать видео")
            return

        self.first_frame = frame.copy()
        self.show_frame(self.first_frame)
        self.set_progress(0)
        self.player.status_label.config(
            text="Выделите мышью прямоугольник терминала"
        )

    def configure_progress(self, total_frames):
        max_frame = max(0, total_frames - 1)
        self.player.progress_scale.config(to=max_frame)
        self.set_progress(0)

    def set_progress(self, frame_index):
        if self.slider_dragging:
            return

        frame_index = self.clamp_frame_index(frame_index)
        self.suppress_slider_command = True
        self.player.progress_var.set(frame_index)
        self.suppress_slider_command = False
        self.update_progress_label(frame_index)

    def update_progress_label(self, frame_index):
        if self.video_total_frames <= 0:
            self.player.progress_label.config(text="0 / 0")
            return

        current = min(self.video_total_frames, frame_index + 1)
        self.player.progress_label.config(
            text=f"{current} / {self.video_total_frames}"
        )

    def clamp_frame_index(self, frame_index):
        if self.video_total_frames <= 0:
            return 0
        return min(max(0, int(float(frame_index))), self.video_total_frames - 1)

    def show_frame(self, frame):
        self.root.update_idletasks()
        label_w = self.player.video_label.winfo_width()
        label_h = self.player.video_label.winfo_height()
        frame_h, frame_w = frame.shape[:2]

        self.display_meta = {
            "frame_w": frame_w,
            "frame_h": frame_h,
            "image_w": frame_w,
            "image_h": frame_h,
            "offset_x": (label_w - frame_w) // 2,
            "offset_y": (label_h - frame_h) // 2,
        }

        imgtk = cv_to_tk(frame)
        self.player.video_label.imgtk = imgtk
        self.player.video_label.configure(image=imgtk)

    def on_zone_press(self, event):
        if self.running or self.first_frame is None:
            return

        point = self.display_to_frame_point(event.x, event.y)
        if point is None:
            return

        self.zone_start = point

    def on_zone_drag(self, event):
        if self.zone_start is None or self.first_frame is None:
            return

        current = self.display_to_frame_point(event.x, event.y, clamp=True)
        if current is None:
            return

        self.draw_zone_preview(self.zone_start, current)

    def on_zone_release(self, event):
        if self.zone_start is None or self.first_frame is None:
            return

        end = self.display_to_frame_point(event.x, event.y, clamp=True)
        if end is None:
            self.zone_start = None
            return

        x1, y1, x2, y2 = self.normalize_rect(self.zone_start, end)
        self.zone_start = None

        if abs(x2 - x1) < 10 or abs(y2 - y1) < 10:
            self.terminal_zone = None
            self.show_frame(self.first_frame)
            self.player.status_label.config(text="Зона слишком маленькая")
            return

        frame_h, frame_w = self.first_frame.shape[:2]
        self.terminal_zone = (
            x1 / frame_w,
            y1 / frame_h,
            x2 / frame_w,
            y2 / frame_h,
        )
        self.draw_zone_preview((x1, y1), (x2, y2))
        self.player.status_label.config(text="Зона терминала выбрана")

    def on_progress_press(self, event):
        self.slider_dragging = True
        self.was_running_before_seek = self.running

        if self.running:
            self.stop(set_status=False)
            self.player.status_label.config(text="Анализ приостановлен")

    def on_progress_change(self, value):
        if self.suppress_slider_command or not self.slider_dragging:
            return

        frame_index = self.clamp_frame_index(value)
        self.update_progress_label(frame_index)
        self.preview_frame_at(frame_index)

    def on_progress_release(self, event):
        if not self.slider_dragging:
            return

        frame_index = self.clamp_frame_index(self.player.progress_var.get())
        self.slider_dragging = False
        self.preview_frame_at(frame_index)
        self.set_progress(frame_index)

        should_resume = self.was_running_before_seek
        self.was_running_before_seek = False
        if should_resume:
            self.start()

    def toggle_playback(self, event=None):
        if self.running:
            self.stop(set_status=False)
            self.player.status_label.config(text="Анализ на паузе")
            return "break"

        self.start()
        return "break"

    def preview_frame_at(self, frame_index):
        if not self.video.cap:
            return

        ret, frame = self.video.read_at(frame_index)
        if not ret:
            return

        self.first_frame = frame.copy()
        self.show_frame(self.frame_with_terminal_zone(frame))

    def frame_with_terminal_zone(self, frame):
        if self.terminal_zone is None:
            return frame

        frame = frame.copy()
        h, w = frame.shape[:2]
        x1 = int(self.terminal_zone[0] * w)
        y1 = int(self.terminal_zone[1] * h)
        x2 = int(self.terminal_zone[2] * w)
        y2 = int(self.terminal_zone[3] * h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 255), 2)
        cv2.putText(
            frame,
            "TERMINAL",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 180, 255),
            2,
            cv2.LINE_AA,
        )
        return frame

    def display_to_frame_point(self, x, y, clamp=False):
        if not self.display_meta:
            return None

        meta = self.display_meta
        image_x = x - meta["offset_x"]
        image_y = y - meta["offset_y"]

        if clamp:
            image_x = min(max(image_x, 0), meta["image_w"] - 1)
            image_y = min(max(image_y, 0), meta["image_h"] - 1)
        elif (
            image_x < 0
            or image_y < 0
            or image_x >= meta["image_w"]
            or image_y >= meta["image_h"]
        ):
            return None

        scale_x = meta["frame_w"] / meta["image_w"]
        scale_y = meta["frame_h"] / meta["image_h"]
        return int(image_x * scale_x), int(image_y * scale_y)

    @staticmethod
    def normalize_rect(start, end):
        x1, y1 = start
        x2, y2 = end
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

    def draw_zone_preview(self, start, end):
        frame = self.first_frame.copy()
        x1, y1, x2, y2 = self.normalize_rect(start, end)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 255), 2)
        cv2.putText(
            frame,
            "TERMINAL",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 180, 255),
            2,
            cv2.LINE_AA,
        )
        self.show_frame(frame)

    def start(self):
        if not self.video.cap or self.running:
            return

        if self.terminal_zone is None:
            self.player.status_label.config(
                text="Сначала выделите прямоугольник терминала"
            )
            return

        frame_index = self.clamp_frame_index(self.player.progress_var.get())
        analyzer = PaymentAnalyzer(terminal_zone=self.terminal_zone)
        self.running = True
        self.video.seek(frame_index)
        self.pipeline = AsyncVideoPipeline(self.video, self.detector, analyzer)
        self.pipeline.start()
        self.player.status_label.config(text="Анализ запущен")

    def stop(self, set_status=True):
        self.running = False

        if self.pipeline:
            self.pipeline.stop()
            self.pipeline = None
        if set_status and hasattr(self, "player"):
            self.player.status_label.config(text="Анализ остановлен")

    def update_ui(self):
        if self.pipeline:
            try:
                frame, result, events, frame_index = (
                    self.pipeline.output_queue.get_nowait()
                )
                self.show_frame(frame)
                self.set_progress(frame_index)

                for event in events:
                    self.player.status_label.config(text=event.message)
                    print(
                        f"[frame {event.frame_index}] "
                        f"ID {event.track_id}: {event.status.value}"
                    )

            except Exception:
                pass

        self.root.after(15, self.update_ui)
