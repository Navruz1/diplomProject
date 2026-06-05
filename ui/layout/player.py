import tkinter as tk

class VideoPlayer(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg="#2b2b2b")

        self.video_label = tk.Label(
            self,
            bg="#1e1e1e",
            bd=0,
            highlightthickness=0,
            padx=0,
            pady=0,
        )
        self.video_label.pack(fill=tk.BOTH, expand=True)

        self.overlay = tk.Frame(self, bg="#202020")
        self.overlay.place(relx=0.5, rely=1.0, anchor="s", relwidth=1.0)
        self.overlay.lift()

        progress = tk.Frame(self.overlay, bg="#202020")
        progress.pack(fill=tk.X, padx=12, pady=(8, 0))

        self.progress_var = tk.IntVar(value=0)
        self.progress_scale = tk.Scale(
            progress,
            from_=0,
            to=0,
            orient=tk.HORIZONTAL,
            variable=self.progress_var,
            showvalue=False,
            bg="#202020",
            fg="white",
            troughcolor="#111111",
            activebackground="#007acc",
            highlightthickness=0,
        )
        self.progress_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.progress_label = tk.Label(
            progress,
            text="0 / 0",
            width=14,
            bg="#202020",
            fg="#f0f0f0",
            anchor="e",
        )
        self.progress_label.pack(side=tk.LEFT, padx=(8, 0))

        controls = tk.Frame(self.overlay, bg="#202020", height=42)
        controls.pack(fill=tk.X, padx=12, pady=(2, 8))

        self.start_btn = tk.Button(
            controls,
            text="▶ Старт",
            bg="#3c3f41",
            fg="white",
        )
        self.start_btn.pack(side=tk.LEFT, padx=(0, 8), pady=4)

        self.stop_btn = tk.Button(
            controls,
            text="■ Стоп",
            bg="#3c3f41",
            fg="white",
        )
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 10), pady=4)

        self.status_label = tk.Label(
            controls,
            text="Готово к анализу",
            bg="#202020",
            fg="#f0f0f0",
            anchor="w",
        )
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
