import threading
import queue
import time

class AsyncVideoPipeline:
    def __init__(self, video_source, detector, max_queue=5):
        self.video = video_source
        self.detector = detector

        self.input_queue = queue.Queue(maxsize=max_queue)
        self.output_queue = queue.Queue(maxsize=max_queue)

        self.running = False

    def start(self):
        self.running = True
        threading.Thread(target=self._reader, daemon=True).start()
        threading.Thread(target=self._worker, daemon=True).start()

    def stop(self):
        self.running = False

    def _reader(self):
        while self.running:
            if not self.input_queue.full():
                ret, frame = self.video.read()
                if not ret:
                    self.running = False
                    break
                self.input_queue.put(frame)
            else:
                time.sleep(0.005)

    def _worker(self):
        while self.running:
            try:
                frame = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # YOLO + tracking
            frame_drawn, result = self.detector.draw(frame)

            # передаём и кадр, и результат
            self.output_queue.put((frame_drawn, result))