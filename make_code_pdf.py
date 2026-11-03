"""Exports a setup page + particle_hands.py as a highlighted PDF."""
import io
import keyword
import os
import sys
import tokenize
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

SRC, OUT = "particle_hands.py", "output/particle_hands_code.pdf"


def find_font():
    candidates = []
    if os.name == "nt":
        candidates.extend([
            r"C:\\Windows\\Fonts\\consola.ttf",
            r"C:\\Windows\\Fonts\\Consola.ttf",
            r"C:\\Windows\\Fonts\\cour.ttf",
            r"C:\\Windows\\Fonts\\lucon.ttf",
            r"C:\\Windows\\Fonts\\CascadiaMono.ttf",
            r"C:\\Windows\\Fonts\\DejaVuSansMono.ttf",
        ])
    else:
        candidates.extend([
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
            "/System/Library/Fonts/MesloLGSNF-Regular.ttf",
            "/System/Library/Fonts/SFMono-Regular.otf",
        ])

    for path in candidates:
        if os.path.exists(path):
            return path

    for root in [Path(sys.prefix), Path(sys.base_prefix), Path.home()]:
        for suffix in [
            "**/DejaVuSansMono.ttf",
            "**/Consola.ttf",
            "**/CascadiaMono.ttf",
            "**/LiberationMono-Regular.ttf",
            "**/SFMono-Regular.otf",
        ]:
            matches = sorted(root.glob(suffix))
            if matches:
                return str(matches[0])

    raise FileNotFoundError(
        "No monospace font found for PDF export. Install DejaVu Sans Mono or Consolas."
    )


FONT = os.environ.get("CODE_PDF_FONT") or find_font()
BG, HEAD = (0.067, 0.071, 0.090), (0.09, 0.095, 0.125)
COL = dict(txt=(0.80, 0.83, 0.90), kw=(0.78, 0.47, 0.87),
           fn=(0.38, 0.69, 0.94), num=(0.82, 0.60, 0.40),
           str=(0.60, 0.76, 0.47), com=(0.42, 0.45, 0.52),
           op=(0.59, 0.63, 0.71), name=(0.90, 0.75, 0.48),
           me=(0.88, 0.42, 0.46), lib=(0.34, 0.71, 0.76))
GOLD, ICE = (0.91, 0.75, 0.43), (0.47, 0.70, 0.96)
FS, LH = 8.2, 11.4
MX, TOP, BOT = 40, 70, 46

SETUP = [
    ("h", "HOW TO SET UP AND RUN"),
    ("p", "Builds the 1080x1920, 30 fps, 30 s reel: particle hands on"),
    ("p", "top, this source code below, with ambient audio."),
    ("s", "1. What you need"),
    ("p", "Python 3.10 or newer, about 10 minutes of CPU time, and"),
    ("p", "internet for the one-time pip install. ffmpeg is bundled"),
    ("p", "(via imageio-ffmpeg) - nothing else to install."),
    ("s", "2. Project files"),
    ("c", "particle_hands.py   the engine (the code in this PDF)"),
    ("c", "make_reel.py        composes video + code panel + audio"),
    ("c", "make_audio.py       original synthesised ambient audio"),
    ("c", "make_code_pdf.py    builds this PDF"),
    ("c", "assets/reference.png   the photo the hands come from"),
    ("c", "requirements.txt    numpy scipy opencv pygame-ce ffmpeg"),
    ("c", "output/             the MP4 and PDF are written here"),
    ("s", "3. Install (Windows)"),
    ("c", "cd creation_touch2"),
    ("c", "py -m venv .venv"),
    ("c", ".venv\\Scripts\\activate"),
    ("c", "pip install -r requirements.txt"),
    ("s", "   Install (macOS / Linux)"),
    ("c", "cd creation_touch2"),
    ("c", "python3 -m venv .venv && source .venv/bin/activate"),
    ("c", "pip install -r requirements.txt"),
    ("s", "4. Render the video"),
    ("c", "python make_reel.py"),
    ("p", "Output: output/creation_touch_particle_hands.mp4"),
    ("p", "(Windows: use  py make_reel.py  if python is not found.)"),
    ("p", "Expect roughly 0.6 s per frame, 900 frames: about 10 min."),
    ("s", "5. Options"),
    ("c", "python make_reel.py --stills 12,23,29.9"),
    ("p", "   saves PNG frames at those seconds (quick preview)."),
    ("c", "python make_reel.py --duration 8"),
    ("p", "   renders only the first 8 seconds as a short test."),
    ("c", "python make_reel.py --no-audio"),
    ("p", "   skips the soundtrack.  --out FILE.mp4 sets the name."),
    ("c", "python make_code_pdf.py"),
    ("p", "   rebuilds this PDF (needs: pip install reportlab)."),
    ("s", "6. If something goes wrong"),
    ("p", "ModuleNotFoundError: activate the venv, rerun pip install."),
    ("p", "Code looks cut off at the right edge: your system lacks a"),
    ("p", "narrow monospace font; install Consolas or DejaVu Sans Mono."),
    ("p", "Too slow: test with --duration 8 or --stills first."),
    ("p", "Changing particle_hands.py changes the code shown in the"),
    ("p", "video automatically; keep lines under about 62 characters."),
    ("s", "7. How it works, in one paragraph"),
    ("p", "The photo is split into a gold and a blue hand. Every bright"),
    ("p", "line pixel becomes a particle (colour from the photo), a soft"),
    ("p", "volume of particles is added, neighbours are linked into"),
    ("p", "lines, then particles stream in, the hands swing together,"),
    ("p", "and the fingertips meet with a spark and light pulse."),
    ("p", "The code panel follows lines that really executed (tracer)."),
]


def setup_page(cv, W, H, total):
    cv.setFillColorRGB(*BG)
    cv.rect(0, 0, W, H, stroke=0, fill=1)
    cv.setFillColorRGB(*HEAD)
    cv.rect(0, H - 46, W, 46, stroke=0, fill=1)
    cv.setFillColorRGB(*GOLD)
    cv.setFont("Mono", 11)
    cv.drawString(MX, H - 29, "README  /  setup")
    cv.setFillColorRGB(*ICE)
    cv.setFont("Mono", 8)
    cv.drawRightString(W - MX, H - 29,
                       f"THE CREATION TOUCH   page 1/{total}")
    cv.setStrokeColorRGB(*GOLD)
    cv.setLineWidth(0.8)
    cv.line(MX, H - 46, W / 2, H - 46)
    cv.setStrokeColorRGB(*ICE)
    cv.line(W / 2, H - 46, W - MX, H - 46)
    y = H - 78
    for kind, text in SETUP:
        if kind == "h":
            cv.setFillColorRGB(*GOLD)
            cv.setFont("Mono", 15)
            cv.drawString(MX, y, text)
            y -= 24
        elif kind == "s":
            y -= 6
            cv.setFillColorRGB(*ICE)
            cv.setFont("Mono", 10)
            cv.drawString(MX, y, text)
            y -= 15
        elif kind == "c":
            cv.setFillColorRGB(0.09, 0.10, 0.14)
            cv.rect(MX, y - 3.5, W - 2 * MX, 13, stroke=0, fill=1)
            cv.setFillColorRGB(*COL["str"])
            cv.setFont("Mono", 8.4)
            cv.drawString(MX + 8, y, text)
            y -= 15
        else:
            cv.setFillColorRGB(*COL["txt"])
            cv.setFont("Mono", 8.4)
            cv.drawString(MX, y, text)
            y -= 12.2
    cv.showPage()


def segments(lines):
    segs = {i: [] for i in range(1, len(lines) + 1)}
    src = "\n".join(lines) + "\n"
    toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    prev = ""
    for k, (tt, s, (sr, sc), (er, ec), _) in enumerate(toks):
        if sr != er or tt in (tokenize.NEWLINE, tokenize.NL,
                              tokenize.INDENT, tokenize.DEDENT,
                              tokenize.ENDMARKER):
            continue
        c = "txt"
        nxt = toks[k + 1].string if k + 1 < len(toks) else ""
        if tt == tokenize.COMMENT:
            c = "com"
        elif tt == tokenize.STRING:
            c = "str"
        elif tt == tokenize.NUMBER:
            c = "num"
        elif tt == tokenize.OP:
            c = "op"
        elif tt == tokenize.NAME:
            if s in ("True", "False", "None"):
                c = "num"
            elif keyword.iskeyword(s):
                c = "kw"
            elif prev in ("def", "class"):
                c = "name"
            elif s in ("np", "cv2"):
                c = "lib"
            elif s == "self":
                c = "me"
            elif nxt == "(":
                c = "fn"
        segs[sr].append((sc, ec, c))
        prev = s
    return segs


def main():
    with open(SRC, encoding="utf-8") as fh:
        lines = fh.read().rstrip("\n").split("\n")
    segs = segments(lines)
    pdfmetrics.registerFont(TTFont("Mono", FONT))
    W, H = A4
    cv = canvas.Canvas(OUT, pagesize=A4)
    cv.setTitle("particle_hands.py - The Creation Touch")
    cv.setAuthor("SKILL CIRCUIT")
    per_page = int((H - TOP - BOT) // LH)
    pages = (len(lines) + per_page - 1) // per_page
    total = pages + 1
    cw = pdfmetrics.stringWidth("M", "Mono", FS)
    setup_page(cv, W, H, total)
    for p in range(pages):
        cv.setFillColorRGB(*BG)
        cv.rect(0, 0, W, H, stroke=0, fill=1)
        cv.setFillColorRGB(*HEAD)
        cv.rect(0, H - 46, W, 46, stroke=0, fill=1)
        cv.setFillColorRGB(*GOLD)
        cv.setFont("Mono", 11)
        cv.drawString(MX, H - 29, "particle_hands.py")
        cv.setFillColorRGB(*ICE)
        cv.setFont("Mono", 8)
        cv.drawRightString(W - MX, H - 29,
                           f"THE CREATION TOUCH   page {p + 2}/{total}")
        cv.setStrokeColorRGB(*GOLD)
        cv.setLineWidth(0.8)
        cv.line(MX, H - 46, W / 2, H - 46)
        cv.setStrokeColorRGB(*ICE)
        cv.line(W / 2, H - 46, W - MX, H - 46)
        y = H - TOP
        for i in range(p * per_page, min(len(lines), (p + 1) * per_page)):
            ln, text = i + 1, lines[i]
            cv.setFont("Mono", FS)
            cv.setFillColorRGB(0.33, 0.36, 0.44)
            cv.drawRightString(MX + 24, y, str(ln))
            x0 = MX + 34
            pos = 0
            for a, b, c in sorted(segs[ln]):
                if a > pos:
                    cv.setFillColorRGB(*COL["txt"])
                    cv.drawString(x0 + pos * cw, y, text[pos:a])
                cv.setFillColorRGB(*COL[c])
                cv.drawString(x0 + a * cw, y, text[a:b])
                pos = b
            if pos < len(text):
                cv.setFillColorRGB(*COL["txt"])
                cv.drawString(x0 + pos * cw, y, text[pos:])
            y -= LH
        cv.showPage()
    cv.save()
    print("saved", OUT, total, "pages")


if __name__ == "__main__":
    main()
