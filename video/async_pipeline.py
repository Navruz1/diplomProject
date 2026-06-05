import queue
import threading
import time

from model.payment_analyzer import PaymentAnalyzer


class AsyncVideoPipeline:
    def __init__(self, video_source, detector, analyzer=None, max_queue=5):
        self.video = video_source
        self.detector = detector
        self.analyzer = analyzer or PaymentAnalyzer()

        self.input_queue = queue.Queue(maxsize=max_queue)
        self.output_queue = queue.Queue(maxsize=max_queue)

        self.running = False
        self.reader_thread = None
        self.worker_thread = None

    def start(self):
        self.analyzer.reset()
        self.running = True
        self.reader_thread = threading.Thread(target=self._reader, daemon=True)
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.reader_thread.start()
        self.worker_thread.start()

    def stop(self, wait=True):
        self.running = False
        if wait:
            self._join_threads()
        self._clear_queue(self.input_queue)
        self._clear_queue(self.output_queue)

    def _join_threads(self):
        current = threading.current_thread()
        for thread in (self.reader_thread, self.worker_thread):
            if thread and thread is not current and thread.is_alive():
                thread.join(timeout=1.5)

    @staticmethod
    def _clear_queue(target_queue):
        while True:
            try:
                target_queue.get_nowait()
            except queue.Empty:
                break

    def _reader(self):
        while self.running:
            if not self.input_queue.full():
                ret, frame = self.video.read()
                if not ret:
                    self.running = False
                    break
                frame_index = max(0, self.video.get_position() - 1)
                self.input_queue.put((frame_index, frame))
            else:
                time.sleep(0.005)

    def _worker(self):
        while self.running:
            try:
                frame_index, frame = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            frame_drawn, result = self.detector.draw(frame)
            events = self.analyzer.analyze(result)
            frame_drawn = self.analyzer.draw_overlay(frame_drawn)

            if self.running:
                self.output_queue.put((frame_drawn, result, events, frame_index))
