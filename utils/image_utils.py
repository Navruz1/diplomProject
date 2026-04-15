import cv2
from PIL import Image, ImageTk

# def cv_to_tk(frame, size=(800, 450)):
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     image = Image.fromarray(frame)
#     image = image.resize(size)
#     return ImageTk.PhotoImage(image=image)

def cv_to_tk(frame, max_size):
    h, w = frame.shape[:2]
    max_w, max_h = max_size

    scale = min(max_w / w, max_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame)
    image = image.resize((new_w, new_h))
    return ImageTk.PhotoImage(image=image)