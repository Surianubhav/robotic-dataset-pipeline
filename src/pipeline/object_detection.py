class ObjectDetector:
    def __init__(self):
        self.model = None
        self.names: dict = {}
        if not YOLO_OK:
            return
        print(f"📦 Loading {Cfg.YOLO_MODEL} ...")
        self.model = _YOLO(Cfg.YOLO_MODEL)
        self.names = self.model.names
        print("✅ YOLO ready.")

    def detect(self, frame: np.ndarray):
        if self.model is None:
            return None
        return self.model(frame, verbose=False, conf=Cfg.YOLO_CONF)[0]

    def result_names(self, results) -> list:
        out = []
        if results is None:
            return out
        for box in results.boxes:
            cls_id = int(box.cls[0])
            name   = self.names.get(cls_id, str(cls_id))
            conf   = float(box.conf[0])
            coords = tuple(map(int, box.xyxy[0].tolist()))
            out.append((name, conf, coords))
        return out
