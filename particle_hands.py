"""THE CREATION TOUCH - particles from the reference"""
import os
import numpy as np
import cv2
from scipy.spatial import cKDTree

W, H, FPS, DURATION = 1080, 1114, 30, 30.0
HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "assets", "reference.png")
TIP = np.array([835.0, 405.0])    # fingertip gap in the photo
CENTRE = np.array([540.0, 557.0])
SCALE = 0.85                      # photo px -> video px
AXIS = np.array([0.90, 0.44])     # gold -> blue direction
N_FILL, N_AIR = 36000, 9000
T_TOUCH = 21.8                    # fingertips meet (seconds)


def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


# --- PHASE 1: the reference becomes data --------------
def load_reference(path=REF):
    bgr = cv2.imread(path)
    return bgr[..., ::-1].astype(np.float32)


def split_hands(rgb):
    # bright, connected areas = hands; a line through the
    # fingertip gap separates gold (left) from ice (right)
    lum = cv2.GaussianBlur(rgb.max(2), (0, 0), 3)
    solid = (lum > 12).astype(np.uint8)
    solid = cv2.morphologyEx(solid, cv2.MORPH_CLOSE,
                             np.ones((15, 15), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(solid)
    big = [i for i in range(1, n) if st[i, 4] > 8000]
    hand = np.isin(lab, big)
    yy, xx = np.mgrid[0:rgb.shape[0], 0:rgb.shape[1]]
    side = (xx - TIP[0]) * AXIS[0] + (yy - TIP[1]) * AXIS[1]
    return hand & (side < 0), hand & (side >= 0), ~hand


# --- PHASE 2: particle initialisation -----------------
def ridge_points(rgb, mask):
    # every pixel of a bright line becomes one particle
    lum = rgb.max(2)
    low = cv2.GaussianBlur(lum, (0, 0), 2.5)
    y, x = np.nonzero(mask & (lum - low > 5) & (lum > 24))
    return np.stack([x, y], 1).astype(np.float32), rgb[y, x]


def fill_points(rng, rgb, mask, n):
    # soft volume: density follows blurred brightness
    low = cv2.GaussianBlur(rgb.max(2), (0, 0), 2.5) / 255.0
    w = np.where(mask, low, 0.0).ravel()
    idx = rng.choice(w.size, n, p=w / w.sum())
    y, x = np.divmod(idx, rgb.shape[1])
    pos = np.stack([x, y], 1) + rng.uniform(-0.5, 0.5, (n, 2))
    return pos, rgb[y, x]


def find_nodes(rgb, mask):
    # luminous joints = local brightness peaks
    lm = cv2.GaussianBlur(rgb.max(2), (0, 0), 2.0)
    top = cv2.dilate(lm, np.ones((31, 31), np.uint8))
    ys, xs = np.nonzero((lm >= top) & (lm > 185) & mask)
    return np.stack([xs, ys], 1).astype(np.float32)


# --- PHASE 3: mesh generation + connections -----------
def build_mesh(pos, n_ridge):
    # join neighbouring line particles into continuous lines
    pts = pos[:n_ridge]
    tree = cKDTree(pts)
    d, nb = tree.query(pts, k=3, distance_upper_bound=2.4)
    a = np.repeat(np.arange(n_ridge), 2)
    b = nb[:, 1:].reshape(-1)
    ok = np.isfinite(d[:, 1:]).reshape(-1)
    pairs = np.stack([a[ok], b[ok]], 1)
    pairs.sort(axis=1)
    return np.unique(pairs, axis=0)


def make_hand(rng, rgb, mask, pivot, away, swing, t0):
    rp, rc = ridge_points(rgb, mask)
    inner = cv2.erode(mask.astype(np.uint8),
                      np.ones((9, 9), np.uint8)) > 0
    fp, fc = fill_points(rng, rgb, inner, N_FILL)
    pos, col = np.vstack([rp, fp]), np.vstack([rc, fc])
    nr, n = len(rp), len(rp) + len(fp)
    soft = cv2.GaussianBlur(rgb.max(2), (0, 0), 2.5)
    low = soft[mask].sum()
    level = col.max(1) * 0.80
    level[nr:] = low * SCALE ** 2 / N_FILL * 0.22
    pivot = np.array(pivot, float)
    along = (pos - pivot) @ (-away)
    nodes = find_nodes(rgb, mask)
    nidx = cKDTree(pos[:nr]).query(nodes)[1]
    return dict(pos=pos, nr=nr, t0=t0, level=level,
                hue=col / (col.max(1, keepdims=True) + 1e-6),
                edges=build_mesh(pos, nr), nidx=nidx,
                pivot=pivot, away=away, swing=swing,
                delay=3.5 * np.clip(along / 900, 0, 1)
                + 0.8 * rng.random(n),
                start=rng.normal(0, 40, (n, 2)),
                ph=rng.uniform(0, 6.28, n),
                freq=rng.uniform(2.0, 6.0, n))


# --- PHASE 3: blue arrival - particles stream in ------
def arrive(h, t, tgt):
    # particles stream in from the arm side, wrist first
    e = smooth((t - h["t0"] - h["delay"]) / 3.0)
    st = tgt - h["away"] * 650.0 + h["start"]
    d = tgt - st
    sw = np.stack([-d[:, 1], d[:, 0]], 1) * 0.12
    pos = st + d * e[:, None]
    pos += sw * np.sin(np.pi * e)[:, None]
    seen = smooth((t - h["t0"] - 0.5 * h["delay"]) / 1.0)
    return pos, e, seen * (0.25 + 0.75 * e)


# --- PHASE 4: reaching - hands swing together ---------
def reach(t):
    # 1 = far apart, 0 = the pose of the photograph
    return 1.0 - smooth((t - 13.0) / 8.8)


def pose(h, t):
    # swing about the arm pivot and slide toward the centre
    r = reach(t)
    a = np.radians(h["swing"] * r)
    c, s = np.cos(a), np.sin(a)
    rot = np.array([[c, -s], [s, c]])
    p = (h["pos"] - h["pivot"]) @ rot.T + h["pivot"]
    return p + h["away"] * 150.0 * r


def to_screen(p, zoom):
    return CENTRE + (p - TIP) * (SCALE * zoom)


def wave(h, t, p):
    # luminous waves run back from the touching fingertip
    k = t - T_TOUCH
    d = np.linalg.norm(p - TIP, axis=1)
    out = np.zeros(len(p))
    for lag in (0.0, 3.0):
        if k - lag > 0:
            front = 300.0 * (k - lag)
            out += np.exp(-((d - front) / 55.0) ** 2)
    return out


# --- PHASE 5: the touch -------------------------------
def touch_fx(t, zoom, ed, glow, pts, cols, S):
    k = t - T_TOUCH
    if k < -1.2:
        return
    mid = to_screen(TIP, zoom)
    power = smooth((k + 1.2) / 1.2)
    power *= 1.0 + 0.7 * np.exp(-k / 1.4)
    add_patch(glow, mid, S["spark"], power * 1.5)
    for i in range(3):
        r = 10 + 240 * (k - 0.5 * i) if k > 0.5 * i else 0
        if 0 < r < 300:
            f = (1 - r / 300) ** 1.5
            halves = (((255, 215, 140), 100),
                      ((190, 225, 255), 280))
            for col, a0 in halves:
                rgb = tuple(float(v * f * 0.7) for v in col)
                cv2.ellipse(ed, (int(mid[0]), int(mid[1])),
                            (int(r), int(r)), 0, a0, a0 + 160,
                            rgb, 1, cv2.LINE_AA)
    if k > 0:
        ph = (S["sp"] + 0.35 * k) % 1.0
        x = (ph - 0.5) * 70 * zoom
        y = 20 * np.sin(np.pi * ph) * S["sy"] * zoom
        p = mid + np.stack([x, y], 1)
        mix = ph[:, None]
        rgb = np.array([255, 225, 160.]) * (1 - mix)
        rgb += np.array([190, 225, 255.]) * mix
        pts.append(p)
        cols.append(rgb * 1.3 * smooth(k / 0.6))


def add_patch(img, c, patch, gain):
    # add a small glow sprite centred on c
    r = patch.shape[0] // 2
    x, y = int(c[0]), int(c[1])
    x0, y0 = max(x - r, 0), max(y - r, 0)
    x1, y1 = min(x + r + 1, W), min(y + r + 1, H)
    if x1 > x0 and y1 > y0:
        sub = patch[y0 - y + r:y1 - y + r,
                    x0 - x + r:x1 - x + r]
        warm = np.array([255.0, 232.0, 185.0])
        img[y0:y1, x0:x1] += sub[..., None] * gain * warm


# --- PHASE 5: glow and rendering ----------------------
def splat(buf, p, rgb):
    # sub-pixel additive splat of tiny particles
    x0 = np.floor(p[:, 0]).astype(int)
    y0 = np.floor(p[:, 1]).astype(int)
    fx, fy = p[:, 0] - x0, p[:, 1] - y0
    taps = ((0, 0, (1 - fx) * (1 - fy)),
            (1, 0, fx * (1 - fy)),
            (0, 1, (1 - fx) * fy),
            (1, 1, fx * fy))
    for dx, dy, w in taps:
        x, y = x0 + dx, y0 + dy
        ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        idx = (y * W + x)[ok]
        for c in range(3):
            wt = (rgb[:, c] * w)[ok]
            buf[:, c] += np.bincount(idx, wt, W * H)


def draw_edges(ed, pa, pb, lv, hue, steps=6):
    # fine AA lines grouped by brightness and colour
    q = np.minimum((lv * steps).astype(int), steps)
    share = hue[:, 2] / (hue.sum(1) + 1e-6)
    blu = np.minimum((share * 4).astype(int), 3)
    for b in range(1, steps + 1):
        for k in range(4):
            m = (q == b) & (blu == k)
            if not m.any():
                continue
            seg = np.stack([pa[m], pb[m]], 1) * 16
            seg = np.round(seg).astype(np.int32)
            col = hue[m].mean(0) * 255 * b / steps
            col = tuple(float(v) for v in col)
            cv2.polylines(ed, seg, False, col, 1,
                          cv2.LINE_AA, 4)


def finish(img):
    # light bloom + hue-preserving tone mapping
    q = (W // 4, H // 4)
    small = cv2.resize(img, q, interpolation=3)
    b1 = cv2.GaussianBlur(small, (0, 0), 2.0)
    b2 = cv2.GaussianBlur(small, (0, 0), 8.0)
    up = lambda b: cv2.resize(b, (W, H))
    out = img + 0.35 * up(b1) + 0.45 * up(b2)
    m = out.max(axis=2, keepdims=True) + 1e-6
    out *= 255.0 * (1.0 - np.exp(-m / 230.0)) / m
    return out.astype(np.uint8)


def draw_hand(h, t, zoom, ed, glow, pts, cols, S):
    pos, e, seen = arrive(h, t, pose(h, t))
    pos = pos + 0.8 * np.sin(0.8 * t + h["ph"])[:, None]
    sc = to_screen(pos, zoom)
    lum = seen * (1 + 2.0 * wave(h, t, pos))
    lum *= 1 + 0.15 * np.sin(h["freq"] * t + h["ph"])
    pts.append(sc)
    cols.append(h["hue"] * (h["level"] * lum)[:, None])
    a, b = h["edges"][:, 0], h["edges"][:, 1]
    lv = 0.5 * (h["level"][a] + h["level"][b]) / 255.0
    lv = lv * np.minimum(e[a], e[b])
    lv = lv * np.minimum(seen[a], seen[b])
    hue = 0.5 * (h["hue"][a] + h["hue"][b])
    draw_edges(ed, sc[a], sc[b], lv * 0.6, hue)
    for j in h["nidx"]:
        gain = 0.7 * seen[j] * e[j]
        add_patch(glow, sc[j], S["node"], gain)


def build(seed=7):
    rng = np.random.default_rng(seed)
    rgb = load_reference()
    gold, ice, air = split_hands(rgb)
    hands = [
        make_hand(rng, rgb, gold, (0, 60), -AXIS, 5.0, 3.0),
        make_hand(rng, rgb, ice, (1672, 700), AXIS, 5.0, 8.4)]
    yy, xx = np.mgrid[-60:61, -60:61].astype(np.float32)
    r2 = xx ** 2 + yy ** 2
    g = lambda s: np.exp(-r2 / (2 * s ** 2))
    spark = g(7.0) + 0.3 * g(24.0)
    node = g(2.2) + 0.12 * g(9.0)
    dust, dcol = ridge_points(rgb, air)
    soft, scol = fill_points(rng, rgb, air, N_AIR)
    apos = np.vstack([dust, soft])
    acol = np.vstack([dcol, scol])
    return dict(hands=hands, air=(apos, acol / 255.0),
                spark=spark.astype(np.float32),
                node=node.astype(np.float32),
                sp=rng.random(90), sy=rng.uniform(-1, 1, 90),
                appear=rng.uniform(0.3, 4.0, len(apos)))


def frame(S, t):
    zoom = 1.0 + 0.22 * smooth((t - 24.5) / 5.0)
    ed = np.zeros((H, W, 3), np.float32)
    glow = np.zeros((H, W, 3), np.float32)
    pts, cols = [], []
    apos, acol = S["air"]
    a = smooth((t - S["appear"]) / 1.5)
    pts.append(to_screen(apos, zoom))
    cols.append(acol * (170.0 * a)[:, None])
    for h in S["hands"]:
        draw_hand(h, t, zoom, ed, glow, pts, cols, S)
    touch_fx(t, zoom, ed, glow, pts, cols, S)
    buf = np.zeros((W * H, 3), np.float32)
    splat(buf, np.vstack(pts), np.vstack(cols))
    img = buf.reshape(H, W, 3) + ed + glow
    return finish(img)
