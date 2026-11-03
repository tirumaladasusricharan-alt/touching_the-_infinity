"""Original ambient pad + touch chime (synthesised, no samples)."""
import wave

import numpy as np

SR = 44100


def synth(duration=30.0, seed=11):
    rng = np.random.default_rng(seed)
    n = int(SR * duration)
    t = np.arange(n) / SR
    out = np.zeros((2, n))
    chord = [(98.0, 1.0), (146.83, 0.7), (196.0, 0.8), (246.94, 0.5),
             (293.66, 0.45), (392.0, 0.25), (587.33, 0.18)]
    for k, (f, a) in enumerate(chord):
        for side in (0, 1):
            det = 1 + rng.uniform(-0.0025, 0.0025)
            lfo = 0.65 + 0.35 * np.sin(2 * np.pi * rng.uniform(0.05, 0.2)
                                       * t + rng.uniform(0, 6.28))
            out[side] += a * lfo * np.sin(2 * np.pi * f * det * t
                                          + rng.uniform(0, 6.28))
    # glittering plucks while the golden wave travels (18-25 s)
    scale = [880.0, 987.77, 1108.7, 1318.5, 1480.0, 1760.0]
    for _ in range(40):
        t0 = rng.uniform(21.8, 28.5)
        f = rng.choice(scale)
        i0 = int(t0 * SR)
        m = min(int(1.6 * SR), n - i0)
        tt = np.arange(m) / SR
        ping = 0.07 * np.sin(2 * np.pi * f * tt) * np.exp(-tt * 3.2)
        pan = rng.uniform(0.2, 0.8)
        out[0, i0:i0 + m] += ping * (1 - pan)
        out[1, i0:i0 + m] += ping * pan
    # soft bell when the fingertips meet (21.8 s)
    i0 = int(21.8 * SR)
    tt = np.arange(min(int(5 * SR), n - i0)) / SR
    bell = sum(a * np.sin(2 * np.pi * f * tt) * np.exp(-tt * d)
               for f, a, d in ((392.0, 0.5, 1.1), (784.0, 0.3, 1.6),
                               (1175.0, 0.2, 2.2), (1568.0, 0.12, 3.0)))
    out[0, i0:i0 + len(tt)] += 0.9 * bell
    out[1, i0:i0 + len(tt)] += 0.9 * bell
    # swell, fade in/out
    env = (0.35 + 0.65 * np.clip((t - 3) / 22, 0, 1)) \
        * np.clip(t / 3.0, 0, 1) * np.clip((duration - t) / 2.5, 0, 1)
    out *= env
    # simple reverb: convolve with decaying noise
    ir_t = np.arange(int(2.8 * SR)) / SR
    for side in (0, 1):
        ir = rng.normal(0, 1, len(ir_t)) * np.exp(-ir_t * 1.7)
        ir[0] = 4.0
        spec = np.fft.rfft(out[side], n + len(ir)) * np.fft.rfft(ir, n + len(ir))
        out[side] = np.fft.irfft(spec)[:n]
    out /= np.abs(out).max() / 0.55
    return out


def write(path="output/ambient.wav"):
    out = synth()
    pcm = (out.T * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return path


if __name__ == "__main__":
    print("wrote", write())
