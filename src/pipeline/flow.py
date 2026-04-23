class FlowAnalyzer:
    def __init__(self):
        self._prev: Optional[np.ndarray] = None

    def reset(self):
        self._prev = None

    def analyze(self, frame: np.ndarray) -> dict:
        gray = cv2.cvtColor(
            cv2.resize(frame, (Cfg.FLOW_W, Cfg.FLOW_H)),
            cv2.COLOR_BGR2GRAY
        )
        out = {"raw_mag": 0.0, "residual_mag": 0.0, "direction": "none"}
        if self._prev is None:
            self._prev = gray
            return out

        flow = cv2.calcOpticalFlowFarneback(
            self._prev, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        out["raw_mag"] = float(np.mean(mag))

        dx = float(np.median(flow[..., 0]))
        dy = float(np.median(flow[..., 1]))
        res = flow.copy()
        res[..., 0] -= dx
        res[..., 1] -= dy
        rmag, rang = cv2.cartToPolar(res[..., 0], res[..., 1])
        out["residual_mag"] = float(np.mean(rmag))

        dirs = ["R", "UR", "U", "UL", "L", "DL", "D", "DR"]
        active = rang[rmag > 0.5]
        if len(active) > 0:
            h, _ = np.histogram(np.degrees(active), bins=8, range=(0, 360))
            out["direction"] = dirs[int(np.argmax(h))]

        self._prev = gray
        return out

