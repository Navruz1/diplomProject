from ultralytics import YOLO

class PersonDetector:
    def __init__(self, model_path="yolov8n.pt", conf=0.4):
        self.model = YOLO(model_path)
        self.conf = conf

    def track(self, frame):
        """
        YOLO + ByteTrack
        persist=True – сохраняет ID между кадрами
        """
        results = self.model.track(
            frame,
            persist=True,
            classes=[0],     # только люди
            conf=self.conf,
            verbose=False
        )
        return results[0]

    def draw(self, frame):
        result = self.track(frame)
        return result.plot(), result