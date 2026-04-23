class Renderer:
    CONF_BAR_W = 130

    def draw_boxes(self, canvas: np.ndarray, detections: list,
                   scale_x: float, scale_y: float):
        for name, conf, (x1, y1, x2, y2) in detections:
            sx1 = int(x1 * scale_x); sy1 = int(y1 * scale_y)
            sx2 = int(x2 * scale_x); sy2 = int(y2 * scale_y)
            color = PAL.get("person" if name == "person" else "default_box")
            cv2.rectangle(canvas, (sx1, sy1), (sx2, sy2), color, 2)
            lbl = f"{name}  {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(lbl, Cfg.FONT, 0.50, 1)
            py = max(sy1 - th - 6, 0)
            cv2.rectangle(canvas, (sx1, py), (sx1 + tw + 8, py + th + 6), color, -1)
            cv2.putText(canvas, lbl, (sx1 + 4, py + th + 2),
                        Cfg.FONT, 0.50, (0, 0, 0), 1, cv2.LINE_AA)

    # Hand bone connections (21 landmarks, works with both MP APIs)
    _HAND_CONN = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17),
    ]

    def draw_hands(self, canvas: np.ndarray, gestures: list,
                   scale_x: float, scale_y: float, infer_w: int, infer_h: int):
        if not gestures:
            return
        conn = self._HAND_CONN
        for g in gestures:
            lm  = g["landmarks"]
            pts = [
                (int(l.x * infer_w * scale_x),
                 int(l.y * infer_h * scale_y))
                for l in lm
            ]
            for c in conn:
                cv2.line(canvas, pts[c[0]], pts[c[1]], (80, 255, 140), 1)
            for pt in pts:
                cv2.circle(canvas, pt, 3, (0, 220, 80), -1)
            wx, wy = pts[0]
            label  = g["gesture"]
            color  = PAL.get(label, PAL["unknown"])
            cv2.putText(canvas, f"{label} ({g['hand']})", (wx + 8, wy - 8),
                        Cfg.FONT_B, 0.55, color, 1, cv2.LINE_AA)

    def draw_hud(self, canvas: np.ndarray,
                 activity: str, act_conf: float,
                 gestures: list, objects: list,
                 flow_mag: float, fps: float, timestamp: float,
                 frame_w: int, frame_h_with_hud: int):

        hud_top = frame_h_with_hud - Cfg.HUD_H
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, hud_top), (frame_w, frame_h_with_hud), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.80, canvas, 0.20, 0, canvas)

        act_color = PAL.get(activity, PAL["unknown"])
        cv2.rectangle(canvas, (0, hud_top), (frame_w, hud_top + 4), act_color, -1)

        act_display = ACTIVITY_DISPLAY.get(activity, activity.replace("_", " ").title())
        cv2.putText(canvas, act_display.upper(),
                    (14, hud_top + 38), Cfg.FONT_B, 0.95, act_color, 2, cv2.LINE_AA)

        bx = frame_w - self.CONF_BAR_W - 14
        by = hud_top + 14
        cv2.rectangle(canvas, (bx, by), (bx + self.CONF_BAR_W, by + 12), (45, 45, 45), -1)
        filled = int(self.CONF_BAR_W * min(act_conf, 1.0))
        cv2.rectangle(canvas, (bx, by), (bx + filled, by + 12), act_color, -1)
        cv2.putText(canvas, f"conf {act_conf:.0%}",
                    (bx, by + 26), Cfg.FONT, 0.42, (160, 160, 160), 1, cv2.LINE_AA)

        if gestures:
            g_parts = []
            for g in gestures[:2]:
                g_parts.append(f"{g['hand']}: {g['gesture']} ({g['conf']:.0%})")
            g_str = "  |  ".join(g_parts)
        else:
            g_str = "No hands detected"

        cv2.putText(canvas, f"Gesture:  {g_str}",
                    (14, hud_top + 64), Cfg.FONT, 0.52, (210, 210, 210), 1, cv2.LINE_AA)

        obj_str = "Objects:  " + ("  ·  ".join(objects[:Cfg.MAX_OBJECTS]) if objects else "—")
        cv2.putText(canvas, obj_str,
                    (14, hud_top + 90), Cfg.FONT, 0.50, (170, 170, 170), 1, cv2.LINE_AA)

        stat_str = f"Flow: {flow_mag:.2f}    FPS: {fps:.1f}    t = {timestamp:.1f}s"
        cv2.putText(canvas, stat_str,
                    (14, hud_top + 116), Cfg.FONT, 0.45, (110, 110, 110), 1, cv2.LINE_AA)

        if flow_mag > Cfg.MOTION_THRESH:
            self._draw_flow_dot(canvas, flow_mag, frame_w - 40, 30)

    @staticmethod
    def _draw_flow_dot(canvas, mag, cx, cy):
        r     = min(int(mag * 4), 24)
        alpha = min(mag / 6.0, 1.0)
        color = (int(50 * (1 - alpha)), int(200 * alpha), int(255 * alpha))
        cv2.circle(canvas, (cx, cy), r, color, -1)
        cv2.putText(canvas, "motion", (cx - 22, cy + r + 14),
                    Cfg.FONT, 0.38, (140, 140, 140), 1, cv2.LINE_AA)

