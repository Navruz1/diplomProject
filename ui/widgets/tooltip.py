import tkinter as tk

def attach_tooltip(widget, text):
    tip = tk.Toplevel(widget)
    tip.withdraw()
    tip.overrideredirect(True)

    label = tk.Label(tip, text=text, bg="#333", fg="white", padx=6)
    label.pack()

    def show(e):
        tip.geometry(f"+{e.x_root+10}+{e.y_root+10}")
        tip.deiconify()

    def hide(e):
        tip.withdraw()

    widget.bind("<Enter>", show)
    widget.bind("<Leave>", hide)