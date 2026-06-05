import cv2
from PIL import Image, ImageTk

def cv_to_tk(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame)
    return ImageTk.PhotoImage(image=image)
