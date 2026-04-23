class GestureRecognizer:
    WRIST       = 0
    THUMB_TIP   = 4;  THUMB_MCP  = 2
    INDEX_TIP   = 8;  INDEX_MCP  = 5
    MIDDLE_TIP  = 12; MIDDLE_MCP = 9
    RING_TIP    = 16; RING_MCP   = 13
    PINKY_TIP   = 20; PINKY_MCP  = 17

    def __init__(self):
        self.hands = None
        self._wrist_hist: deque = deque(maxlen=Cfg.WRIST_WINDOW)
        self._gesture_hist: deque = deque(maxlen=12)
        self._use_new_api = False
        self._frame_ts_ms = 0
        # Store HAND_CONNECTIONS for drawing (set after API init)
        self._hand_connections = None
        if not MP_OK:
            return
        # ── Try legacy solutions API (mediapipe < 0.10.x) ─────────
        if hasattr(_mp, "solutions") and hasattr(_mp.solutions, "hands"):
            _h = _mp.solutions.hands
            self.hands = _h.Hands(
                static_image_mode        = False,
                max_num_hands            = Cfg.MAX_HANDS,
                min_detection_confidence = Cfg.HAND_CONF,
                min_tracking_confidence  = Cfg.HAND_TRACK_CONF,
            )
            self._hand_connections = _mp.solutions.hands.HAND_CONNECTIONS
            print("✅ MediaPipe (legacy solutions API) ready.")
        # ── New Tasks API (mediapipe >= 0.10.14) ───────────────────
        else:
            try:
                from mediapipe.tasks.python import vision as _mp_vision
                from mediapipe.tasks.python import BaseOptions as _mpBO
                import urllib.request, os as _os
                model_path = _os.path.join(
                    _os.path.dirname(_os.path.abspath(__file__)),
                    "hand_landmarker.task"
                )
                if not _os.path.exists(model_path):
                    url = ("https://storage.googleapis.com/mediapipe-models/"
                           "hand_landmarker/hand_landmarker/float16/latest/"
                           "hand_landmarker.task")
                    print("📥 Downloading hand_landmarker.task (~9 MB) ...")
                    urllib.request.urlretrieve(url, model_path)
                    print("✅ Model saved to", model_path)
                opts = _mp_vision.HandLandmarkerOptions(
                    base_options=_mpBO(model_asset_path=model_path),
                    running_mode=_mp_vision.RunningMode.VIDEO,
                    num_hands=Cfg.MAX_HANDS,
                    min_hand_detection_confidence=Cfg.HAND_CONF,
                    min_tracking_confidence=Cfg.HAND_TRACK_CONF,
                )
                self.hands = _mp_vision.HandLandmarker.create_from_options(opts)
                self._use_new_api = True
                # New API uses integer pair tuples for connections
                self._hand_connections = [
                    (0,1),(1,2),(2,3),(3,4),
                    (0,5),(5,6),(6,7),(7,8),
                    (0,9),(9,10),(10,11),(11,12),
                    (0,13),(13,14),(14,15),(15,16),
                    (0,17),(17,18),(18,19),(19,20),
                    (5,9),(9,13),(13,17),
                ]
                print("✅ MediaPipe (new Tasks API) ready.")
            except Exception as e:
                print(f"⚠  MediaPipe Tasks API failed: {e}")
                print("   Gestures disabled. YOLO + activity still works.")
                self.hands = None

    def _extended(self, lm, tip, mcp, axis="y") -> bool:
        if axis == "y":
            return lm[tip].y < lm[mcp].y
        else:
            return abs(lm[tip].x - lm[self.WRIST].x) > abs(lm[mcp].x - lm[self.WRIST].x)

    def _finger_states(self, lm) -> dict:
        return {
            "thumb":  self._extended(lm, self.THUMB_TIP,  self.THUMB_MCP,  axis="x"),
            "index":  self._extended(lm, self.INDEX_TIP,  self.INDEX_MCP),
            "middle": self._extended(lm, self.MIDDLE_TIP, self.MIDDLE_MCP),
            "ring":   self._extended(lm, self.RING_TIP,   self.RING_MCP),
            "pinky":  self._extended(lm, self.PINKY_TIP,  self.PINKY_MCP),
        }

    def _static_gesture(self, fs: dict) -> str:
        t, i, m, r, p = fs["thumb"], fs["index"], fs["middle"], fs["ring"], fs["pinky"]
        n_ext = sum([t, i, m, r, p])

        if n_ext == 0:                                              return "fist"
        if n_ext == 5:                                              return "open_palm"
        if i and m and not r and not p:                             return "peace"
        if i and not m and not r and not p:                         return "point"
        if t and not i and not m and not r and not p:               return "thumbs_up"
        if t and i and not m and not r and not p:                   return "ok"
        if t and not i and not m and not r and p:                   return "call_me"
        if not t and i and not m and not r and p:                   return "rock"
        return "idle"

    def _dynamic_gesture(self) -> Optional[str]:
        if len(self._wrist_hist) < 10:
            return None
        pts     = np.array(self._wrist_hist)
        h_range = float(np.ptp(pts[:, 0]))
        v_range = float(np.ptp(pts[:, 1]))
        net_disp = float(np.linalg.norm(pts[-1] - pts[0]))
        if h_range > 0.15 and h_range > v_range * 1.2 and net_disp < 0.15:
            return "wave"
        return None

    def process(self, frame: np.ndarray) -> tuple:
        if self.hands is None:
            return [], None

        if self._use_new_api:
            return self._process_new_api(frame)
        else:
            return self._process_legacy(frame)

    def _process_legacy(self, frame: np.ndarray) -> tuple:
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        gestures = []

        if not results or not results.multi_hand_landmarks:
            self._wrist_hist.clear()
            self._gesture_hist.append("idle")
            return [], results

        for idx, hand_lm in enumerate(results.multi_hand_landmarks):
            lm = hand_lm.landmark
            hand_label = "R"
            if results.multi_handedness and idx < len(results.multi_handedness):
                hand_label = results.multi_handedness[idx].classification[0].label[0]

            wrist_pt = np.array([lm[self.WRIST].x, lm[self.WRIST].y])
            if idx == 0:
                self._wrist_hist.append(wrist_pt)

            fs        = self._finger_states(lm)
            static_g  = self._static_gesture(fs)
            dynamic_g = self._dynamic_gesture()
            final_g   = dynamic_g if dynamic_g else static_g

            self._gesture_hist.append(final_g)
            conf = Counter(self._gesture_hist).most_common(1)[0][1] / len(self._gesture_hist)

            gestures.append({
                "gesture":   final_g,
                "conf":      round(conf, 2),
                "hand":      hand_label,
                "fingers":   fs,
                "landmarks": lm,
            })

        return gestures, results

    def _process_new_api(self, frame: np.ndarray) -> tuple:
        """Handle mediapipe >= 0.10.14 HandLandmarker Tasks API."""
        import mediapipe as _mp2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = _mp2.Image(image_format=_mp2.ImageFormat.SRGB, data=rgb)
        self._frame_ts_ms += 33   # ~30 fps timestamp increment
        results = self.hands.detect_for_video(mp_image, self._frame_ts_ms)
        gestures = []

        if not results or not results.hand_landmarks:
            self._wrist_hist.clear()
            self._gesture_hist.append("idle")
            return [], None

        for idx, hand_lm in enumerate(results.hand_landmarks):
            # New API: hand_lm is a list of NormalizedLandmark objects
            hand_label = "R"
            if results.handedness and idx < len(results.handedness):
                hand_label = results.handedness[idx][0].display_name[0]

            wrist_pt = np.array([hand_lm[self.WRIST].x, hand_lm[self.WRIST].y])
            if idx == 0:
                self._wrist_hist.append(wrist_pt)

            fs        = self._finger_states(hand_lm)
            static_g  = self._static_gesture(fs)
            dynamic_g = self._dynamic_gesture()
            final_g   = dynamic_g if dynamic_g else static_g

            self._gesture_hist.append(final_g)
            conf = Counter(self._gesture_hist).most_common(1)[0][1] / len(self._gesture_hist)

            gestures.append({
                "gesture":   final_g,
                "conf":      round(conf, 2),
                "hand":      hand_label,
                "fingers":   fs,
                "landmarks": hand_lm,   # list of NormalizedLandmark
            })

        return gestures, None

