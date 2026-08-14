#!/usr/bin/env python3
"""
Generate a Y4M video with a physiologically-shaped pulse baked into the pixels,
for end-to-end testing of the browser capture path via Chromium's fake camera.

Per-pixel spatial noise is included on purpose: a perfectly uniform frame would
quantise identically in every pixel, turning the sub-1% pulse into a staircase.
Real sensors dither, and averaging thousands of pixels is what recovers the
signal — the test has to reproduce that or it isn't testing the real mechanism.
"""
import numpy as np, sys, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--bpm", type=float, default=72)
ap.add_argument("--sec", type=float, default=30)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--w", type=int, default=160)
ap.add_argument("--h", type=int, default=120)
ap.add_argument("--mode", choices=["face", "finger"], default="face")
ap.add_argument("--melanin", type=float, default=0.0)
ap.add_argument("--out", required=True)
a = ap.parse_args()

n = int(a.sec * a.fps)
f = a.bpm / 60.0
rng = np.random.default_rng(7)

if a.mode == "finger":
    base = np.array([205.0, 62.0, 34.0])      # flash-lit fingertip
    amp = np.array([7.0, 10.0, 4.0])          # strong contact-PPG modulation
else:
    atten = (1 - 0.75 * a.melanin) ** 1.5
    dc = 1 - 0.72 * a.melanin
    base = np.array([168.0, 128.0, 112.0]) * dc
    amp = np.array([0.55, 1.75, 1.00]) * atten  # hemoglobin: green modulates most

with open(a.out, "wb") as fh:
    fh.write(("YUV4MPEG2 W%d H%d F%d:1 Ip A1:1 C420jpeg\n" % (a.w, a.h, a.fps)).encode())
    for i in range(n):
        t = i / a.fps
        p = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(4 * np.pi * f * t + 0.6)
        drift = 1.2 * np.sin(2 * np.pi * 0.05 * t)
        rgb = base + amp * p + drift

        img = np.empty((a.h, a.w, 3), dtype=np.float64)
        img[:, :, 0] = rgb[0]; img[:, :, 1] = rgb[1]; img[:, :, 2] = rgb[2]
        img += rng.normal(0, 3.5, img.shape)          # spatial sensor noise -> dithering
        img = np.clip(img, 0, 255)

        R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        Y = 0.299 * R + 0.587 * G + 0.114 * B
        U = -0.168736 * R - 0.331264 * G + 0.5 * B + 128
        V = 0.5 * R - 0.418688 * G - 0.081312 * B + 128

        fh.write(b"FRAME\n")
        fh.write(np.clip(Y, 0, 255).astype(np.uint8).tobytes())
        fh.write(np.clip(U[::2, ::2], 0, 255).astype(np.uint8).tobytes())
        fh.write(np.clip(V[::2, ::2], 0, 255).astype(np.uint8).tobytes())

print("wrote %s  (%d frames, %.1f bpm, %s mode, melanin %.2f)" % (a.out, n, a.bpm, a.mode, a.melanin))
