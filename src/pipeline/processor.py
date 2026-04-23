from .flow import FlowAnalyzer
from .object_detection import ObjectDetector
from .gestures import GestureRecognizer
from .activity import ActivityClassifier
from src.utils.visualization import Renderer

class FrameProcessor:
    def __init__(self):
        # ── Instantiate all sub-modules here ──────────────────────
        self.flow_az    = FlowAnalyzer()
        self.detector   = ObjectDetector()
        self.gesture_r  = GestureRecognizer()
        self.activity_c = ActivityClassifier()
        self.renderer   = Renderer()

        # State caches
        self._last_yolo     = None
        self._last_objs     = []
        self._last_gestures = []
        self._last_flow     = {"raw_mag": 0.0, "residual_mag": 0.0, "direction": "none"}

        self._fps        = 0.0
        self._t_last     = time.time()
        self._infer_count = 0

    def process(self, frame: np.ndarray,
                infer_w: int, infer_h: int,
                run_inference: bool) -> dict:
        # Always run flow (cheap)
        flow_info = self.flow_az.analyze(frame)

        if run_inference:
            small = cv2.resize(frame, (infer_w, infer_h))

            yolo_res = self.detector.detect(small)
            self._last_yolo = yolo_res
            self._last_objs = self.detector.result_names(yolo_res)

            gestures, _ = self.gesture_r.process(small)
            self._last_gestures = gestures

            self.activity_c.update(self._last_objs, flow_info, infer_w, infer_h)

            now = time.time()
            self._fps = 1.0 / max(now - self._t_last, 1e-4)
            self._t_last = now
            self._infer_count += 1

        else:
            self.activity_c.update(self._last_objs, flow_info, infer_w, infer_h)

        self._last_flow = flow_info
        return self._build_state()

    def _build_state(self) -> dict:
        dom_objs = self._dominant_objects()
        return {
            "activity":  self.activity_c.current_activity,
            "act_conf":  self.activity_c.confidence,
            "gestures":  self._last_gestures,
            "objects":   dom_objs,
            "flow":      self._last_flow,
            "yolo_raw":  self._last_yolo,
            "fps":       self._fps,
        }

    def _dominant_objects(self) -> list:
        counts = self.activity_c._obj_hist
        obj_c: Counter = Counter()
        for frame_objs in counts:
            for o in frame_objs:
                obj_c[o] += 1
        n = max(len(counts), 1)
        return [o for o, c in obj_c.most_common(Cfg.MAX_OBJECTS)
                if c / n >= Cfg.OBJ_FREQ_THRESH]

    def render(self, frame: np.ndarray, state: dict,
               infer_w: int, infer_h: int, timestamp: float) -> np.ndarray:
        h, w = frame.shape[:2]
        canvas_h = h + Cfg.HUD_H
        canvas = np.zeros((canvas_h, w, 3), dtype=np.uint8)
        canvas[:h] = frame

        scale_x = w / infer_w
        scale_y = h / infer_h

        # FIX: draw boxes from cached detections, not a broken conditional
        raw_detections = self.detector.result_names(state["yolo_raw"])
        self.renderer.draw_boxes(canvas, raw_detections, scale_x, scale_y)
        self.renderer.draw_hands(canvas, state["gestures"],
                                 scale_x, scale_y, infer_w, infer_h)
        self.renderer.draw_hud(
            canvas,
            activity         = state["activity"],
            act_conf         = state["act_conf"],
            gestures         = state["gestures"],
            objects          = state["objects"],
            flow_mag         = state["flow"]["residual_mag"],
            fps              = state["fps"],
            timestamp        = timestamp,
            frame_w          = w,
            frame_h_with_hud = canvas_h,
        )
        return canvas

def run_video(input_path: str, output_path: str,
              infer_every: int = None, preview: bool = False):
    if not os.path.exists(input_path):
        print(f"❌ File not found: {input_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Cannot open: {input_path}")
        sys.exit(1)

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_h   = src_h + Cfg.HUD_H

    step = infer_every or max(1, int(src_fps / Cfg.TARGET_FPS))

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        src_fps,
        (src_w, out_h)
    )

    proc    = FrameProcessor()
    t_start = time.time()

    print(f"\n🎬 Annotating: {input_path}")
    print(f"   {src_w}×{src_h} @ {src_fps:.1f}fps  |  {total_f} frames")
    print(f"   Inference every {step} frames  |  Output → {output_path}\n")

    for idx in range(total_f):
        ret, frame = cap.read()
        if not ret:
            break

        run_inf   = (idx % step == 0)
        timestamp = idx / src_fps
        state     = proc.process(frame, Cfg.INFER_W, Cfg.INFER_H, run_inf)
        canvas    = proc.render(frame, state, Cfg.INFER_W, Cfg.INFER_H, timestamp)

        writer.write(canvas)

        if preview:
            cv2.imshow("Preview (Q=quit)", canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if idx % 150 == 0:
            elapsed = time.time() - t_start
            pct     = idx / max(total_f, 1) * 100
            eta     = (elapsed / max(idx, 1)) * (total_f - idx)
            print(f"  [{pct:5.1f}%]  frame {idx:>6}/{total_f}  "
                  f"activity={state['activity']:<14}  "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s")

    cap.release()
    writer.release()
    if preview:
        cv2.destroyAllWindows()
    print(f"\n✅ Saved → {output_path}")

def run_live(source):
    print("🚀 Starting live mode...")

    if isinstance(source, str) and source.isdigit():
        source = int(source)

    # Try default backend first, fall back to DSHOW on Windows
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("⚠  Retrying with DirectShow backend...")
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print(f"❌ Cannot open camera/source: {source}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step    = max(1, int(src_fps / Cfg.TARGET_FPS))

    proc    = FrameProcessor()
    t_start = time.time()
    idx     = 0

    win = "Activity Recognizer — Press Q to quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print("✅ Camera opened. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame — exiting.")
            break

        try:
            run_inf   = (idx % step == 0)
            timestamp = time.time() - t_start
            state     = proc.process(frame, Cfg.INFER_W, Cfg.INFER_H, run_inf)
            canvas    = proc.render(frame, state, Cfg.INFER_W, Cfg.INFER_H, timestamp)
        except Exception as e:
            print(f"⚠️  Frame {idx} error: {e}")
            idx += 1
            continue

        cv2.imshow(win, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

        idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print("👋 Live session ended.")

