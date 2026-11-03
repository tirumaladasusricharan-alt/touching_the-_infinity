"""Composites the particle hands (top 58 %) and the REAL source
code (bottom 42 %) into output/creation_touch_particle_hands.mp4.

The line tracer records which lines of particle_hands.py really
executed; the editor panel follows those lines.
"""
import argparse
import ast
import io
import keyword
import math
import os
import subprocess
import sys
import tokenize

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import numpy as np
import pygame

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "particle_hands.py")
OUT = os.path.join(HERE, "output")
TOP_H, W, H = 1114, 1080, 1920          # 58 % / 42 % split
FPS = 30
ICE = (120, 178, 245)
GOLD = (232, 190, 110)
DIM_GOLD = (150, 122, 70)
LH, FONT_PX, GUTTER = 38, 25, 84
RATE = 30.0                              # lines typed per second
COL = dict(kw=(198, 120, 221), fn=(97, 175, 239), num=(209, 154, 102),
           str=(152, 195, 121), com=(106, 114, 132), txt=(196, 202, 214),
           op=(150, 160, 180), name=(229, 192, 123), me=(224, 108, 117),
           lib=(86, 182, 194))
# (start time, [function names]) - in file order, one block per phase beat
SEQUENCE = [
    (0.0, ["__header__"]), (1.0, ["smooth", "load_reference"]),
    (2.0, ["split_hands"]), (3.2, ["ridge_points"]),
    (4.6, ["fill_points"]), (5.9, ["find_nodes"]),
    (7.0, ["build_mesh"]), (8.6, ["make_hand"]),
    (10.2, ["arrive"]), (13.0, ["reach", "pose"]),
    (16.5, ["to_screen", "wave"]), (20.6, ["touch_fx"]),
    (23.0, ["add_patch", "splat"]), (24.4, ["draw_edges"]),
    (25.4, ["finish"]), (26.2, ["draw_hand"]),
    (27.8, ["build"]), (28.8, ["frame"]),
]
PHASES = [(0.0, "01  DARKNESS"), (3.0, "02  GOLDEN ARRIVAL"),
          (8.4, "03  BLUE ARRIVAL"), (13.0, "04  REACHING"),
          (21.8, "05  THE TOUCH"), (25.0, "06  HERO SHOT")]


# ------------------------------------------------------------ tracing
class Tracer:
    """Records the real line numbers executed inside SRC."""

    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.events = []
        self.seen = set()

    def _glob(self, frame, event, arg):
        if os.path.abspath(frame.f_code.co_filename) == self.path:
            return self._loc
        return None

    def _loc(self, frame, event, arg):
        if event == "line":
            self.events.append(frame.f_lineno)
        return self._loc

    def call(self, fn, *a):
        prev = sys.gettrace()
        sys.settrace(self._glob)
        try:
            return fn(*a)
        finally:
            sys.settrace(prev)
            self.seen.update(self.events)
            self.events = []


# ------------------------------------------------------- code layout
def partition(lines):
    """Split the file into blocks; every line belongs to one block."""
    tree = ast.parse("\n".join(lines))
    starts = []
    first_def = min(n.lineno for n in tree.body
                    if isinstance(n, ast.FunctionDef))
    for node in tree.body:
        name = None
        if isinstance(node, ast.FunctionDef):
            name = node.name
        elif isinstance(node, ast.Assign) and node.lineno > first_def:
            name = node.targets[0].id
        if name:
            a = node.lineno
            while a > 1 and lines[a - 2].lstrip().startswith("#"):
                a -= 1
            starts.append((name, a))
    blocks = {"__header__": (1, starts[0][1] - 1)}
    for i, (name, a) in enumerate(starts):
        b = starts[i + 1][1] - 1 if i + 1 < len(starts) else len(lines)
        blocks[name] = (a, b)
    return blocks


def build_groups(blocks):
    groups = []
    for t0, names in SEQUENCE:
        lo = min(blocks[n][0] for n in names)
        hi = max(blocks[n][1] for n in names)
        groups.append((t0, lo, hi))
    return groups


def highlight(lines):
    segs = {i: [] for i in range(1, len(lines) + 1)}
    toks = list(tokenize.generate_tokens(io.StringIO("\n".join(lines) + "\n").readline))
    prev = ""
    for k, (tt, s, (sr, sc), (er, ec), _) in enumerate(toks):
        if sr != er or tt in (tokenize.NEWLINE, tokenize.NL, tokenize.INDENT,
                              tokenize.DEDENT, tokenize.ENDMARKER):
            continue
        c = COL["txt"]
        nxt = toks[k + 1].string if k + 1 < len(toks) else ""
        if tt == tokenize.COMMENT:
            c = COL["com"]
        elif tt == tokenize.STRING:
            c = COL["str"]
        elif tt == tokenize.NUMBER:
            c = COL["num"]
        elif tt == tokenize.OP:
            c = COL["op"]
        elif tt == tokenize.NAME:
            if s in ("True", "False", "None"):
                c = COL["num"]
            elif keyword.iskeyword(s):
                c = COL["kw"]
            elif prev in ("def", "class"):
                c = COL["name"]
            elif s in ("np", "cv2"):
                c = COL["lib"]
            elif s == "self":
                c = COL["me"]
            elif nxt == "(":
                c = COL["fn"]
        segs[sr].append((sc, ec, c))
        prev = s
    out = {}
    for ln, text in enumerate(lines, 1):
        parts, pos = [], 0
        for a, b, c in sorted(segs[ln]):
            if a > pos:
                parts.append((text[pos:a], COL["txt"]))
            parts.append((text[a:b], c))
            pos = b
        if pos < len(text):
            parts.append((text[pos:], COL["txt"]))
        out[ln] = parts
    return out


class CodePanel:
    def __init__(self, lines, tracer):
        self.lines, self.tracer = lines, tracer
        self.blocks = partition(lines)
        self.groups = build_groups(self.blocks)
        self.font = pygame.font.SysFont(
            "consolas,cascadiamono,dejavusansmono,couriernew,monospace",
            FONT_PX)
        self.surfs = self._render()
        self.cw = self.font.size("M")[0]
        self.ed = pygame.Rect(22, TOP_H + 20, W - 44, H - TOP_H - 42)
        self.head_h = 62
        self.area = pygame.Rect(self.ed.x + 2, self.ed.y + self.head_h,
                                self.ed.w - 4, self.ed.h - self.head_h - 6)
        self.scroll = 0.0
        self.bands = {}
        self.label_font = pygame.font.SysFont("segoeui,helveticaneue,dejavusans,arial", 22, bold=True)

    def _render(self):
        parts = highlight(self.lines)
        out = []
        for ln in range(1, len(self.lines) + 1):
            w = max(1, sum(self.font.size(t)[0] for t, _ in parts[ln]))
            s = pygame.Surface((w, LH), pygame.SRCALPHA)
            x = 0
            for text, c in parts[ln]:
                if text.strip():
                    s.blit(self.font.render(text, True, c),
                           (x, (LH - self.font.get_height()) // 2))
                x += self.font.size(text)[0]
            out.append(s)
        return out

    def _band(self, w, color, alpha):
        key = (w, color, alpha)
        if key not in self.bands:
            s = pygame.Surface((w, LH), pygame.SRCALPHA)
            s.fill((*color, alpha))
            self.bands[key] = s
        return self.bands[key]

    def state(self, t):
        """-> active group, typed-line count, cursor line."""
        gi = max(i for i, g in enumerate(self.groups) if g[0] <= t + 1e-6)
        t0, lo, hi = self.groups[gi]
        typed = (t - t0) * RATE
        done = typed >= hi - lo + 1
        code = [n for n in range(lo, hi + 1) if n in self.tracer.seen]
        cur = None
        if code:
            cur = code[int((t - t0) * 3.5) % len(code)]
        return gi, typed, done, cur

    def draw(self, surf, t, dt):
        gi, typed, done, cur = self.state(t)
        _, lo, hi = self.groups[gi]
        # --- editor chrome
        surf.fill((8, 8, 11), (0, TOP_H, W, H - TOP_H))
        pygame.draw.rect(surf, (17, 18, 23), self.ed, border_radius=20)
        pygame.draw.rect(surf, (74, 62, 38), self.ed, 2, border_radius=20)
        head = pygame.Rect(self.ed.x, self.ed.y, self.ed.w, self.head_h)
        pygame.draw.rect(surf, (23, 24, 31), head, border_top_left_radius=20,
                         border_top_right_radius=20)
        for k, c in enumerate(((255, 96, 96), (255, 190, 70), (70, 210, 120))):
            pygame.draw.circle(surf, c, (head.x + 36 + k * 28, head.centery), 8)
        name = self.label_font.render("particle_hands.py", True, GOLD)
        surf.blit(name, (head.x + 140, head.centery - name.get_height() // 2))
        pygame.draw.line(surf, GOLD, (head.x + 140, head.bottom - 4),
                         (head.x + 140 + name.get_width(), head.bottom - 4), 2)
        label = [p[1] for p in PHASES if p[0] <= t][-1]
        lab = self.label_font.render(label, True, ICE)
        surf.blit(lab, (head.right - 28 - lab.get_width(),
                        head.centery - lab.get_height() // 2))
        # --- smooth scroll target
        n_in = hi - lo + 1
        view = self.area.h
        top_lim = (lo - 1) * LH - LH * 0.5
        if n_in * LH <= view * 0.9:
            # keep earlier code visible above the active block
            target = min(top_lim, hi * LH - view * 0.92)
        elif not done:
            target = (lo - 1 + typed) * LH - view * 0.72
        else:
            target = (cur or lo) * LH - view * 0.40
        target = max(0.0, min(target, len(self.lines) * LH - view * 0.55))
        self.scroll += (target - self.scroll) * (1 - math.exp(-dt * 6.0))
        clip = surf.get_clip()
        surf.set_clip(self.area)
        first = max(0, int(self.scroll // LH))
        for i in range(first, min(len(self.lines), first + view // LH + 2)):
            ln = i + 1
            # typed fraction of this line
            gline = None
            frac = 1.0
            for k, (g0, glo, ghi) in enumerate(self.groups):
                if glo <= ln <= ghi:
                    frac = max(0.0, min(1.0, (t - g0) * RATE - (ln - glo)))
                    break
            if frac <= 0:
                continue
            y = self.area.y + i * LH - self.scroll
            active = lo <= ln <= hi
            if active:
                surf.blit(self._band(self.area.w, GOLD, 16), (self.area.x, y))
            if ln == cur and done:
                surf.blit(self._band(self.area.w, GOLD, 58), (self.area.x, y))
                surf.fill(GOLD, (self.area.x, y, 5, LH))
            elif active:
                surf.fill((120, 98, 56), (self.area.x, y, 3, LH))
            num = self.font.render(str(ln), True,
                                   GOLD if (ln == cur and done) else (84, 88, 102))
            surf.blit(num, (self.area.x + GUTTER - 20 - num.get_width(),
                            y + (LH - num.get_height()) // 2))
            s = self.surfs[i]
            w = int(s.get_width() * frac) if frac < 1 else s.get_width()
            s.set_alpha(255 if active else 130)
            surf.blit(s, (self.area.x + GUTTER, y),
                      pygame.Rect(0, 0, min(w, self.area.w - GUTTER - 10), LH))
        surf.set_clip(clip)
        # fade the editor edges softly
        for k in range(14):
            a = 150 - k * 11
            fade = pygame.Surface((self.area.w, 1), pygame.SRCALPHA)
            fade.fill((17, 18, 23, max(a, 0)))
            surf.blit(fade, (self.area.x, self.area.y + k))
            surf.blit(fade, (self.area.x, self.area.bottom - 1 - k))
        # divider between the two worlds (gold -> ice blue)
        for x in range(60, W - 60, 4):
            k = (x - 60) / (W - 120)
            c = tuple(int(GOLD[q] * (1 - k) + ICE[q] * k) for q in range(3))
            pygame.draw.line(surf, c, (x, TOP_H), (x + 4, TOP_H), 2)
            for g in range(1, 7):
                gl = tuple(int(v * (0.22 - g * 0.03)) for v in c)
                pygame.draw.line(surf, gl, (x, TOP_H - g), (x + 4, TOP_H - g))


# ------------------------------------------------------- composition
def load_engine(tracer):
    """Import particle_hands under the tracer (module-level lines)."""
    import importlib
    sys.path.insert(0, HERE)
    sys.modules.pop("particle_hands", None)
    gb = tracer.call(importlib.import_module, "particle_hands")
    S = tracer.call(gb.build)
    return gb, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(OUT, "creation_touch_particle_hands.mp4"))
    ap.add_argument("--stills", default="", help="comma list of seconds -> PNGs")
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--no-audio", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    pygame.display.set_mode((64, 64))
    with open(SRC, encoding="utf-8") as fh:
        lines = fh.read().split("\n")
    if lines[-1] == "":
        lines.pop()
    tracer = Tracer(SRC)
    gb, S = load_engine(tracer)
    assert (gb.W, gb.H, gb.FPS) == (W, TOP_H, FPS)
    panel = CodePanel(lines, tracer)
    canvas = pygame.Surface((W, H), depth=32)
    top = pygame.Surface((W, TOP_H), depth=32)
    duration = args.duration or gb.DURATION
    total = int(round(duration * FPS))
    stills = {int(round(float(s) * FPS)) for s in args.stills.split(",") if s}
    video = None
    if not stills:
        import imageio_ffmpeg
        silent = os.path.join(OUT, "_silent.mp4")
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
               "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264",
               "-preset", "slow", "-crf", "15", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", silent]
        video = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    dt = 1.0 / FPS
    for i in range(total):
        t = i * dt
        img = tracer.call(gb.frame, S, t)            # RGB uint8 (H, W, 3)
        pygame.surfarray.blit_array(top, img.transpose(1, 0, 2))
        canvas.blit(top, (0, 0))
        panel.draw(canvas, t, dt)
        if i in stills:
            p = os.path.join(OUT, f"still_{t:05.2f}.png")
            pygame.image.save(canvas, p)
            print("saved", p)
        if video:
            video.stdin.write(pygame.image.tobytes(canvas, "RGB"))
            if i % 30 == 0:
                print(f"\rframe {i + 1}/{total}", end="", flush=True)
    if not video:
        return
    video.stdin.close()
    if video.wait() != 0:
        raise SystemExit("ffmpeg failed")
    print()
    final = args.out
    if args.no_audio:
        os.replace(silent, final)
    else:
        import imageio_ffmpeg
        import make_audio
        wav = make_audio.write(os.path.join(OUT, "ambient.wav"))
        subprocess.check_call([imageio_ffmpeg.get_ffmpeg_exe(), "-y",
                               "-loglevel", "error", "-i", silent, "-i", wav,
                               "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                               "-shortest", "-movflags", "+faststart", final])
        os.remove(silent)
    print("saved", final)


if __name__ == "__main__":
    main()
