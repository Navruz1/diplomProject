import tkinter as tk

from ui.widgets.tooltip import attach_tooltip


class Sidebar(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, width=220, bg="#252526")
        self.pack_propagate(False)

        self.add_button = tk.Button(
            self,
            text="+",
            font=("Arial", 18),
            bg="#2d2d30",
            fg="white",
            relief=tk.FLAT,
        )
        self.add_button.pack(pady=(10, 5))

        self.path_label = tk.Label(
            self,
            text="",
            bg="#252526",
            fg="#cccccc",
            anchor="w",
            justify="left",
            wraplength=200,
        )
        self.path_label.pack(fill="x", padx=8, pady=(0, 5))
        attach_tooltip(self.path_label, "")

        self.video_list = tk.Listbox(
            self,
            bg="#1e1e1e",
            fg="white",
            selectbackground="#007acc",
            relief=tk.FLAT,
        )
        self.video_list.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def set_folder_path(self, path: str):
        max_len = 28
        display = path
        if len(path) > max_len:
            display = "..." + path[-(max_len - 3):]

        self.path_label.config(text=display)
        attach_tooltip(self.path_label, path)
