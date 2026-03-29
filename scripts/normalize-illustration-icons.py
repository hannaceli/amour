#!/usr/bin/env python3
"""
Per-image frame removal: detect paper vs foreground, nested mats, dark ink lines,
wood/textured borders, and thin rims — then square crop and resize to 512×512.

Run from repo root: python3 scripts/normalize-illustration-icons.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
ILLUSTRATIONS = ROOT / "public" / "illustrations"
OUT_SIZE = 512
MAX_INNER_ITERS = 5


@dataclass(frozen=True)
class CropProfile:
    max_iters: int = MAX_INNER_ITERS
    prefer_largest_bbox: bool = False
    rim_max_shrink: int = 48
    skip_strip_edges: bool = False
    # No mask / rim / strip — only square + resize (use when auto-crop removed real art).
    passthrough: bool = False


# Per-file tweaks when auto-crop eats real artwork (e.g. dark legs / mane at edges).
FILE_PROFILES: dict[str, CropProfile] = {
    "wc-more-horse2.jpg": CropProfile(passthrough=True),
    # Getting There — keep a bit more margin than default tight bbox + rim shrink.
    "wc-plane.jpg": CropProfile(
        max_iters=2,
        prefer_largest_bbox=True,
        rim_max_shrink=0,
        skip_strip_edges=True,
    ),
    "wc-car.jpg": CropProfile(
        max_iters=2,
        prefer_largest_bbox=True,
        rim_max_shrink=0,
        skip_strip_edges=True,
    ),
}


def luminance(rgb: np.ndarray) -> np.ndarray:
    return np.dot(rgb[..., :3], np.array([0.299, 0.587, 0.114], dtype=np.float64))


def edge_ring(rgb: np.ndarray, thickness: int = 2) -> np.ndarray:
    h, w = rgb.shape[:2]
    t = min(thickness, h // 2, w // 2)
    if t < 1:
        return rgb.reshape(-1, 3)
    parts = [
        rgb[:t, :, :].reshape(-1, 3),
        rgb[-t:, :, :].reshape(-1, 3),
        rgb[:, :t, :].reshape(-1, 3),
        rgb[:, -t:, :].reshape(-1, 3),
    ]
    return np.vstack(parts)


def estimate_paper_color(ring_rgb: np.ndarray) -> np.ndarray:
    r = ring_rgb.reshape(-1, 3)
    lum = np.dot(r, np.array([0.299, 0.587, 0.114]))
    cutoff = np.percentile(lum, 55)
    bright = r[lum >= cutoff]
    if len(bright) < 8:
        bright = r
    return np.median(bright, axis=0).astype(np.float32)


def is_likely_frame_row(row: np.ndarray) -> bool:
    """Wood strips, uniform dark bands, thin ink lines (mixed median)."""
    med = float(np.median(row))
    p5 = float(np.percentile(row, 5))
    p95 = float(np.percentile(row, 95))
    std = float(np.std(row))
    dark_frac = float(np.mean(row < 68.0))
    if med < 68.0 and std < 22.0:
        return True
    if med < 92.0 and p5 < 58.0:
        return True
    if p5 < 48.0 and med < 135.0:
        return True
    if dark_frac > 0.11 and med < 145.0:
        return True
    if med < 118.0 and std > 32.0 and p5 < 95.0:
        return True
    if p95 - p5 < 28.0 and med < 75.0:
        return True
    # Textured wood / decorative strip (e.g. wine bottle card)
    if 78.0 < med < 175.0 and std > 24.0 and dark_frac > 0.06:
        return True
    return False


def is_likely_frame_col(col: np.ndarray) -> bool:
    return is_likely_frame_row(col)


def strip_frame_edges(gray: np.ndarray, max_scan: int = 120) -> tuple[int, int, int, int]:
    h, w = gray.shape
    top, bottom, left, right = 0, h - 1, 0, w - 1
    while top < min(max_scan, h // 2 - 1) and is_likely_frame_row(gray[top]):
        top += 1
    while bottom > top and h - 1 - bottom < max_scan and is_likely_frame_row(gray[bottom]):
        bottom -= 1
    while left < min(max_scan, w // 2 - 1) and is_likely_frame_col(gray[:, left]):
        left += 1
    while right > left and w - 1 - right < max_scan and is_likely_frame_col(gray[:, right]):
        right -= 1
    return top, left, bottom + 1, right + 1


def foreground_mask(
    rgb: np.ndarray,
    paper: np.ndarray,
    color_tol: float,
    lum_ink: float = 52.0,
) -> np.ndarray:
    rgbf = rgb.astype(np.float32)
    d = np.sqrt(np.sum((rgbf - paper.reshape(1, 1, 3)) ** 2, axis=2))
    lum = luminance(rgb)
    ink = lum < lum_ink
    return (d > color_tol) | ink


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1


def clean_mask(mask: np.ndarray) -> np.ndarray:
    u8 = (mask.astype(np.uint8) * 255)
    im = Image.fromarray(u8)
    im = im.filter(ImageFilter.MaxFilter(3))
    im = im.filter(ImageFilter.MinFilter(3))
    return np.array(im) > 128


def shrink_bbox_past_rims(
    gray: np.ndarray,
    bb: tuple[int, int, int, int],
    max_shrink: int = 48,
) -> tuple[int, int, int, int]:
    """
    Walk inward from each side while the outer 2px band still looks like a rim:
    dark ink, brown wood, grey mat, or near-white inner mat.
    """
    yt0, xt0, yt1, xt1 = bb

    def rim_is_border_like(band: np.ndarray) -> bool:
        if band.size == 0:
            return False
        med = float(np.median(band))
        p20 = float(np.percentile(band, 20))
        dark_frac = float(np.mean(band < 72.0))
        # Dark line / wood / grey
        if med < 120.0:
            return True
        if dark_frac > 0.07 and med < 155.0:
            return True
        # Near-white mat inside a double frame
        if med > 232.0 and p20 > 200.0:
            return True
        return False

    steps = 0
    while steps < max_shrink and yt1 - yt0 > 36 and xt1 - xt0 > 36:
        moved = False
        if yt0 + 3 < yt1:
            band = gray[yt0 : yt0 + 2, xt0:xt1]
            if rim_is_border_like(band):
                yt0 += 1
                moved = True
                steps += 1
        if not moved and yt1 - 3 > yt0:
            band = gray[yt1 - 2 : yt1, xt0:xt1]
            if rim_is_border_like(band):
                yt1 -= 1
                moved = True
                steps += 1
        if not moved and xt0 + 3 < xt1:
            band = gray[yt0:yt1, xt0 : xt0 + 2]
            if rim_is_border_like(band):
                xt0 += 1
                moved = True
                steps += 1
        if not moved and xt1 - 3 > xt0:
            band = gray[yt0:yt1, xt1 - 2 : xt1]
            if rim_is_border_like(band):
                xt1 -= 1
                moved = True
                steps += 1
        if not moved:
            break

    return yt0, xt0, yt1, xt1


def square_center_crop(im: Image.Image) -> Image.Image:
    w, h = im.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    return im.crop((left, top, left + s, top + s))


def one_pass_crop(arr: np.ndarray, profile: CropProfile) -> np.ndarray | None:
    """Return cropped array or None if should keep as-is."""
    gray = luminance(arr)
    if profile.skip_strip_edges:
        h, w = gray.shape
        t0, l0, t1, l1 = 0, 0, h, w
    else:
        t0, l0, t1, l1 = strip_frame_edges(gray)
    arr = arr[t0:t1, l0:l1]
    gray = gray[t0:t1, l0:l1]
    h, w = arr.shape[:2]
    if h < 16 or w < 16:
        return None

    ring = edge_ring(arr, thickness=max(2, min(h, w) // 64))
    paper = estimate_paper_color(ring)

    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for color_tol in (50, 42, 34, 28, 22, 16):
        m = foreground_mask(arr, paper, color_tol=color_tol)
        m = clean_mask(m)
        bb = mask_bbox(m)
        if bb is None:
            continue
        yt0, xt0, yt1, xt1 = bb
        area = (yt1 - yt0) * (xt1 - xt0)
        cov = area / float(h * w)
        if cov < 0.016 or cov > 0.992:
            continue
        candidates.append((area, bb))

    if not candidates:
        m = foreground_mask(arr, paper, color_tol=26, lum_ink=68)
        bb = mask_bbox(m)
        if bb is None:
            return arr
        best = bb
    else:
        if profile.prefer_largest_bbox:
            candidates.sort(key=lambda x: x[0], reverse=True)
        else:
            candidates.sort(key=lambda x: x[0])
        best = candidates[0][1]

    yt0, xt0, yt1, xt1 = shrink_bbox_past_rims(gray, best, max_shrink=profile.rim_max_shrink)
    return arr[yt0:yt1, xt0:xt1]


def process_file(path: Path) -> None:
    im = Image.open(path).convert("RGB")
    profile = FILE_PROFILES.get(path.name, CropProfile())

    if profile.passthrough:
        out = square_center_crop(im)
        out = out.resize((OUT_SIZE, OUT_SIZE), Image.Resampling.LANCZOS)
        out.save(path, quality=92, optimize=True)
        return

    arr = np.array(im)
    prev_area = arr.shape[0] * arr.shape[1]
    for _ in range(profile.max_iters):
        nxt = one_pass_crop(arr, profile)
        if nxt is None:
            break
        area = nxt.shape[0] * nxt.shape[1]
        if area >= prev_area * 0.998:
            break
        prev_area = area
        arr = nxt

    out = square_center_crop(Image.fromarray(arr))
    out = out.resize((OUT_SIZE, OUT_SIZE), Image.Resampling.LANCZOS)
    out.save(path, quality=92, optimize=True)


def main() -> int:
    if not ILLUSTRATIONS.is_dir():
        print("Missing", ILLUSTRATIONS, file=sys.stderr)
        return 1
    if len(sys.argv) > 1:
        jpgs = []
        for name in sys.argv[1:]:
            p = ILLUSTRATIONS / name
            if not p.is_file():
                print("Skip (not found):", p, file=sys.stderr)
                continue
            jpgs.append(p)
        if not jpgs:
            return 1
    else:
        jpgs = sorted(ILLUSTRATIONS.glob("*.jpg"))
    if not jpgs:
        print("No JPEGs in", ILLUSTRATIONS, file=sys.stderr)
        return 1
    for p in jpgs:
        try:
            process_file(p)
            print("OK", p.name)
        except Exception as e:
            print("FAIL", p.name, e, file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
