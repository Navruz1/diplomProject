import threading

import cv2


class VideoSource:
    def __init__(self):
        self.cap = None
        self.lock = threading.RLock()

    def open(self, source):
        with self.lock:
            self.cap = cv2.VideoCapture(source)

    def read(self):
        with self.lock:
            if self.cap:
                return self.cap.read()
        return False, None

    def release(self):
        with self.lock:
            if self.cap:
                self.cap.release()

    def get_size(self):
        with self.lock:
            if self.cap:
                return (
                    int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                )
        return 0, 0

    def get_frame_count(self):
        with self.lock:
            if self.cap:
                return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return 0

    def get_position(self):
        with self.lock:
            if self.cap:
                return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        return 0

    def read_first_frame(self):
        return self.read_at(0)

    def read_at(self, frame_index):
        with self.lock:
            if self.cap:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_index)))
                return self.cap.read()
        return False, None

    def seek(self, frame_index):
        with self.lock:
            if self.cap:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_index)))

    def rewind(self):
        self.seek(0)
