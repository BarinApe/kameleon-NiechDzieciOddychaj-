import tkinter as tk
from gui import AplikacjaFizjo


if __name__ == "__main__":
    root = tk.Tk()
    app = AplikacjaFizjo(root)
    root.protocol("WM_DELETE_WINDOW", app.zamknij)
    root.mainloop()