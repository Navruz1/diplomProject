import tkinter as tk

class RootWindow(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Bus Fare Control")
        self.geometry("1200x700")
        self.minsize(900, 600)

        self.configure(bg="#2b2b2b")
