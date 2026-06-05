from pathlib import Path
import threading

from ultralytics import YOLO


class PersonDetector:
    def __init__(self, model_path=None, conf=0.25, tracker_path="model/bus_tracker.yaml"):
        self.model_path = model_path or self._find_default_model()
        self.model = YOLO(self.model_path)
        self.conf = conf
        self.tracker_path = tracker_path
        self.lock = threading.Lock()

    @staticmethod
    def _find_default_model():
        for path in ("model/yolov8n-pose.pt", "yolov8n-pose.pt"):
            if Path(path).exists():
                return path
        return "yolov8n.pt"

    def track(self, frame):
        """
        YOLO + ByteTrack. If a pose model is available, wrist keypoints are
        returned in the same result and used by the payment analyzer.
        persist=True keeps the same object IDs between frames.
        """
        with self.lock:
            results = self.model.track(
                frame,
                persist=True,
                tracker=self.tracker_path,
                classes=[0],  # people only
                conf=self.conf,
                iou=0.5,
                verbose=False,
            )
        return results[0]

    def draw(self, frame):
        result = self.track(frame)
        return frame.copy(), result
