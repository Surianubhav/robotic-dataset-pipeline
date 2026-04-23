class ActivityClassifier:
    ACTIVITY_OBJECTS = {
        "cooking":    {"bowl", "cup", "fork", "spoon", "knife", "bottle",
                       "oven", "refrigerator", "sink", "toaster", "microwave"},
        "phone_use":  {"cell phone"},
        "exercising": {"sports ball", "tennis racket", "baseball bat",
                       "skateboard", "bicycle", "dumbbell"},
        "typing":     {"laptop", "keyboard", "computer", "mouse"},
        "driving":    {"car", "truck", "bus"},
    }

    def __init__(self):
        self._obj_hist   : deque = deque(maxlen=Cfg.SMOOTH_FRAMES)
        self._flow_hist  : deque = deque(maxlen=Cfg.SMOOTH_FRAMES)
        self._person_bb  : deque = deque(maxlen=Cfg.SMOOTH_FRAMES)
        self._label_hist : deque = deque(maxlen=Cfg.SMOOTH_FRAMES)

        self.current_activity = "idle"
        self.confidence       = 0.0

    def update(self, objects: list, flow_info: dict, frame_w: int, frame_h: int) -> str:
        obj_names = [o[0] for o in objects]
        self._obj_hist.append(obj_names)
        self._flow_hist.append(flow_info.get("residual_mag", 0.0))

        person_areas = [
            (x2 - x1) * (y2 - y1) / (frame_w * frame_h)
            for name, conf, (x1, y1, x2, y2) in objects
            if name == "person"
        ]
        self._person_bb.append(person_areas[0] if person_areas else 0.0)

        label = self._classify()
        self._label_hist.append(label)

        smoothed = Counter(self._label_hist).most_common(1)[0][0]
        self.current_activity = smoothed
        dom_count = Counter(self._label_hist).most_common(1)[0][1]
        self.confidence = round(dom_count / len(self._label_hist), 2)
        return smoothed

    def _classify(self) -> str:
        all_objs: Counter = Counter()
        for frame_objs in self._obj_hist:
            for o in frame_objs:
                all_objs[o] += 1
        n = max(len(self._obj_hist), 1)
        dominant = {o for o, c in all_objs.items() if c / n >= Cfg.OBJ_FREQ_THRESH}

        avg_flow     = float(np.mean(self._flow_hist)) if self._flow_hist else 0.0
        avg_person_a = float(np.mean(self._person_bb)) if self._person_bb else 0.0

        for activity, obj_set in self.ACTIVITY_OBJECTS.items():
            if dominant & obj_set:
                return activity

        if "person" not in dominant:
            return "idle"

        if avg_flow > 4.0:   return "running"
        if avg_flow > 1.8:   return "walking"
        if avg_person_a > 0.25 and avg_flow < 1.0:
            return "sitting"
        if avg_flow > 0.5:   return "unknown"

        return "idle"

