# The Creation Touch - particle hands rebuilt from the reference photo

1080x1920, 30 fps, 30 s. Top 58 %: golden and icy-blue hands made of particles and line networks,
reconstructed from `assets/reference.png`. Bottom 42 %: the real `particle_hands.py` in a code editor.

## Build
```
pip install -r requirements.txt
python make_reel.py                    # -> output/creation_touch_particle_hands.mp4
python make_reel.py --stills 12,23,29  # PNG frames for inspection
python make_reel.py --no-audio
```
Windows: use `py`. ffmpeg comes with `imageio-ffmpeg`. About 0.6 s per frame on CPU.

## How it works (image-driven, not a 3D mesh)
* `split_hands` finds the two hands in the reference (bright connected areas, split along the fingertip gap).
* `ridge_points` turns every bright line pixel into one particle, colour taken from the photo;
  `fill_points` adds a soft volume of particles whose density follows brightness.
* `build_mesh` links neighbouring particles (cKDTree) into continuous lines; `find_nodes` finds the bright joints.
* Animation: particles stream in from each arm side, then the hands swing together; at the end the pose
  equals the photo. Touch: spark, pulse rings, particles travelling tip to tip, light waves through the meshes.
* Rendering: sub-pixel additive splatting, anti-aliased lines, light bloom, hue-preserving tone map.

Because the hands come from the photo, their anatomy is the photo's, but they are 2D reconstructions:
the fingers do not move independently, and the hands move as whole shapes.

## Code panel
The video imports `particle_hands.py` under `sys.settrace`, records executed line numbers, types each block
in when its phase begins, highlights it, and moves the active-line marker only across lines that really ran.
