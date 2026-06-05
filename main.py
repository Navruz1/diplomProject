from ui.layout.root_window import RootWindow
from ui.app import AppController

def main():
    root = RootWindow()
    AppController(root)
    root.mainloop()

if __name__ == "__main__":
    main()