import cv2

class VideoSource:
    def __init__(self):
        self.cap = None

    def open(self, source):
        self.cap = cv2.VideoCapture(source)

    def read(self):
        if self.cap:
            return self.cap.read()
        return False, None

    def release(self):
        if self.cap:
            self.cap.release()

    def get_size(self):
        if self.cap:
            return (
                int(self.cap.get(3)),   # cv2.CAP_PROP_FRAME_WIDTH
                int(self.cap.get(4))    # cv2.CAP_PROP_FRAME_HEIGHT
            )

    def read_first_frame(self):
        if self.cap:
            self.cap.set(1, 0)  # cv2.CAP_PROP_POS_FRAMES = 1
            return self.cap.read()
        return False, None