import tkinter as tk
from ui.app import YOLOVideoApp

def main():
    root = tk.Tk()
    app = YOLOVideoApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()