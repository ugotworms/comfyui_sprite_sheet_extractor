"""
pixel_snapper.py — grid-snapping + K-Means quantization ported from the
Sprite Fusion Pixel Snapper (original Rust, MIT © 2025 Hugo Duprez).

Ported into this package so the sprite-sheet workflow can use the same
grid-detection and palette-quantization machinery directly, without the
external `comfyui-spritefusion-pixel-snapper` custom node.

The exposed helpers are:

  * `kmeans_palette(pixels, k, seed, max_iter)` — K-Means++ palette
    extraction, returning (k, 3) float32 centroids. Used by the new
    palette-lock logic in nodes.py.

  * `snap_to_palette(frame, palette)` — nearest-neighbour recolour of a
    whole frame to a fixed (K, 3) palette.

  * `apply_pixel_snap(frame, config)` — the full snapper pipeline on a
    single (H, W, 3) uint8 frame: K-Means quantize → gradient profiles
    → step estimate → walk → stabilize → resample. Returns a snapped
    uint8 image whose cell count differs from the input dimensions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SnapConfig:
    """Tunable parameters for the pixel-snap pipeline. Defaults mirror the
    upstream Sprite Fusion Pixel Snapper Rust defaults."""
    k_colors: int = 16
    pixel_size_override: float = 0.0
    k_seed: int = 42
    max_kmeans_iterations: int = 15
    peak_threshold_multiplier: float = 0.2
    peak_distance_filter: int = 4
    walker_search_window_ratio: float = 0.35
    walker_min_search_window: float = 2.0
    walker_strength_threshold: float = 0.5
    min_cuts_per_axis: int = 4
    fallback_target_segments: int = 64
    max_step_ratio: float = 1.8  # Lowered from 3.0 to catch more skew cases


class PixelSnapperError(ValueError):
    """Raised when invalid input or processing failure occurs."""


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_image_dimensions(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise PixelSnapperError("Image dimensions cannot be zero")
    if width > 10_000 or height > 10_000:
        raise PixelSnapperError("Image dimensions too large (max 10000x10000)")


def upscale_nearest(img: np.ndarray, scale: int) -> np.ndarray:
    """
    Pixel-art-safe integer upscaling by nearest neighbor.
    img: (H, W, 3) uint8
    scale: int >= 1
    """
    if scale <= 1:
        return img
    return np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)


# ---------------------------------------------------------------------------
# K-Means palette extraction (shared core)
# ---------------------------------------------------------------------------

def kmeans_palette(
    pixels: np.ndarray,
    k: int,
    seed: int = 42,
    max_iter: int = 15,
) -> np.ndarray:
    """
    K-Means++ colour quantization on an (N, 3) float32 pixel array.

    Returns the learned centroids as a (K_actual, 3) float32 array
    (K_actual <= k when N < k). This is the quantizer reused by both the
    palette-lock logic and the snap pipeline so the two stay consistent.

    Mirrors the upstream Rust logic: KMeans++ seeding, max_iter Lloyd
    updates, early stop on centroid movement < 0.01.
    """
    if k <= 0:
        raise PixelSnapperError("Number of colors must be greater than 0")

    pixels = np.ascontiguousarray(pixels, dtype=np.float32)
    n_pixels = pixels.shape[0]
    if n_pixels == 0:
        raise PixelSnapperError("No pixels to quantize")

    k = min(k, n_pixels)
    rng = np.random.default_rng(seed)

    # Subsample for centroid learning. Palette quality is essentially
    # unchanged well below full resolution, and this bounds peak memory
    # regardless of frame size — a 768² frame has ~590k pixels, and the
    # (N, K, 3) distance broadcast would exhaust RAM at that scale.
    max_samples = 10000
    if n_pixels > max_samples:
        idx = rng.choice(n_pixels, size=max_samples, replace=False)
        work = pixels[idx]
    else:
        work = pixels
    n_work = work.shape[0]

    def dist_sq(points: np.ndarray, centroid: np.ndarray) -> np.ndarray:
        diff = points - centroid
        return np.sum(diff * diff, axis=1)

    first_idx = int(rng.integers(0, n_work))
    centroids = [work[first_idx]]
    distances = np.full(n_work, np.inf, dtype=np.float32)

    # KMeans++ style init (mirroring Rust logic)
    for _ in range(1, k):
        last_c = centroids[-1]
        d_sq = dist_sq(work, last_c)
        distances = np.minimum(distances, d_sq)
        sum_sq = float(np.sum(distances))
        if sum_sq <= 0.0 or not math.isfinite(sum_sq):
            idx = int(rng.integers(0, n_work))
        else:
            probs = distances / sum_sq
            idx = int(rng.choice(n_work, p=probs))
        centroids.append(work[idx])

    centroids = np.stack(centroids, axis=0)
    prev_centroids = centroids.copy()

    for iteration in range(max_iter):
        # Compute distances to centroids on the (subsampled) working set.
        diff = work[:, None, :] - centroids[None, :, :]
        dists = np.sum(diff * diff, axis=2)
        labels = np.argmin(dists, axis=1)
        del diff, dists

        sums = np.zeros_like(centroids)
        np.add.at(sums, labels, work)
        counts = np.bincount(labels, minlength=k).astype(np.float32)
        counts_expanded = counts[:, None]
        nonzero = counts_expanded[:, 0] > 0
        centroids[nonzero] = sums[nonzero] / counts_expanded[nonzero]

        if iteration > 0:
            movement = np.max(np.sum((centroids - prev_centroids) ** 2, axis=1))
            if movement < 0.01:
                break
        prev_centroids = centroids.copy()

    return centroids


def _assign_nearest(
    pixels: np.ndarray, centroids: np.ndarray, chunk: int = 8192
) -> np.ndarray:
    """
    Nearest-neighbour recolour of (N, 3) pixels to (K, 3) centroids.
    Returns (N, 3) uint8. Processed in chunks so the (chunk, K, 3)
    distance broadcast stays bounded regardless of N.
    """
    n = pixels.shape[0]
    out = np.empty((n, 3), dtype=np.uint8)
    # Pre-convert centroid table to uint8 once; avoids three per-chunk
    # float32 allocations (round + clip + astype) inside the hot loop.
    centroids_u8 = centroids.round().clip(0, 255).astype(np.uint8)
    for s in range(0, n, chunk):
        c = pixels[s : s + chunk]
        diff = c[:, None, :] - centroids[None, :, :]
        dists = np.sum(diff * diff, axis=2)
        labels = np.argmin(dists, axis=1)
        del diff, dists  # free (chunk, K, 3) and (chunk, K) temporaries immediately
        out[s : s + chunk] = centroids_u8[labels]
    return out


def quantize_image(img: np.ndarray, config: SnapConfig) -> np.ndarray:
    """
    Quantize an (H, W, 3) uint8 image to `config.k_colors` colours via
    K-Means and return the recoloured (H, W, 3) uint8 image.
    """
    pixels = img.reshape(-1, 3).astype(np.float32)
    centroids = kmeans_palette(
        pixels, config.k_colors, config.k_seed, config.max_kmeans_iterations
    )
    recoloured = _assign_nearest(pixels, centroids)
    return recoloured.reshape(img.shape)


def snap_to_palette(frame: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """
    Recolour an (H, W, 3) uint8 frame to the nearest colour in `palette`
    ((K, 3) float32 centroids). Returns (H, W, 3) uint8. Chunking is
    handled inside _assign_nearest so peak memory stays bounded for
    large frames.
    """
    H, W, _ = frame.shape
    px = frame.reshape(-1, 3).astype(np.float32)
    snapped = _assign_nearest(px, palette)
    return snapped.reshape(H, W, 3)


# ---------------------------------------------------------------------------
# Grid-detection pipeline (profile → walk → stabilize → resample)
# ---------------------------------------------------------------------------

def compute_profiles(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w, _ = img.shape
    if w < 3 or h < 3:
        raise PixelSnapperError("Image too small (minimum 3x3)")

    gray = (
        0.299 * img[:, :, 0].astype(np.float32)
        + 0.587 * img[:, :, 1].astype(np.float32)
        + 0.114 * img[:, :, 2].astype(np.float32)
    )

    col_proj = np.zeros(w, dtype=np.float32)
    row_proj = np.zeros(h, dtype=np.float32)

    col_proj[1:-1] = np.sum(np.abs(gray[:, 2:] - gray[:, :-2]), axis=0)
    row_proj[1:-1] = np.sum(np.abs(gray[2:, :] - gray[:-2, :]), axis=1)

    return col_proj, row_proj


def estimate_step_size(profile: Sequence[float], config: SnapConfig) -> Optional[float]:
    if len(profile) == 0:
        return None
    max_val = float(np.max(profile))
    if max_val == 0.0:
        return None
    threshold = max_val * config.peak_threshold_multiplier

    peaks: List[int] = []
    for i in range(1, len(profile) - 1):
        if (
            profile[i] > threshold
            and profile[i] > profile[i - 1]
            and profile[i] > profile[i + 1]
        ):
            peaks.append(i)
    if len(peaks) < 2:
        return None

    clean_peaks = [peaks[0]]
    for p in peaks[1:]:
        if p - clean_peaks[-1] > (config.peak_distance_filter - 1):
            clean_peaks.append(p)
    if len(clean_peaks) < 2:
        return None

    diffs = np.diff(clean_peaks)
    diffs.sort()
    return float(diffs[len(diffs) // 2])


def resolve_step_sizes(
    step_x_opt: Optional[float],
    step_y_opt: Optional[float],
    width: int,
    height: int,
    config: SnapConfig,
) -> Tuple[float, float]:
    if config.pixel_size_override != 0.0:
        return config.pixel_size_override, config.pixel_size_override

    if step_x_opt is not None and step_y_opt is not None:
        sx, sy = step_x_opt, step_y_opt
        ratio = sx / sy if sx > sy else sy / sx
        if ratio > config.max_step_ratio:
            smaller = min(sx, sy)
            return smaller, smaller
        avg = (sx + sy) / 2.0
        return avg, avg

    if step_x_opt is not None:
        return step_x_opt, step_x_opt
    if step_y_opt is not None:
        return step_y_opt, step_y_opt

    fallback = (
        (min(width, height) / float(config.fallback_target_segments))
        if config.fallback_target_segments
        else 1.0
    )
    return max(fallback, 1.0), max(fallback, 1.0)


def walk(
    profile: Sequence[float], step_size: float, limit: int, config: SnapConfig
) -> List[int]:
    if len(profile) == 0:
        raise PixelSnapperError("Cannot walk on empty profile")

    cuts = [0]
    current_pos = 0.0
    search_window = max(
        step_size * config.walker_search_window_ratio, config.walker_min_search_window
    )
    mean_val = float(np.mean(profile))

    while current_pos < limit:
        target = current_pos + step_size
        if target >= limit:
            cuts.append(limit)
            break

        start_search = max(int(target - search_window), int(current_pos + 1))
        end_search = min(int(target + search_window), limit)

        if end_search <= start_search:
            current_pos = target
            continue

        segment = np.asarray(profile[start_search:end_search])
        max_rel = int(np.argmax(segment))
        max_val = float(segment[max_rel]) if segment.size else -1.0
        max_idx = start_search + max_rel

        if max_val > mean_val * config.walker_strength_threshold:
            cuts.append(max_idx)
            current_pos = float(max_idx)
        else:
            cuts.append(int(target))
            current_pos = target
    return cuts


def sanitize_cuts(cuts: List[int], limit: int) -> List[int]:
    if limit == 0:
        return [0]

    normalized = []
    has_zero = False
    has_limit = False
    for v in cuts:
        v = 0 if v < 0 else v
        if v == 0:
            has_zero = True
        if v >= limit:
            v = limit
            has_limit = True
        normalized.append(v)
    if not has_zero:
        normalized.append(0)
    if not has_limit:
        normalized.append(limit)

    normalized = sorted(set(normalized))
    return normalized


def snap_uniform_cuts(
    profile: Sequence[float],
    limit: int,
    target_step: float,
    config: SnapConfig,
    min_required: int,
) -> List[int]:
    if limit == 0:
        return [0]
    if limit == 1:
        return [0, 1]

    desired_cells = (
        int(round(limit / target_step))
        if target_step > 0 and math.isfinite(target_step)
        else 0
    )
    desired_cells = max(desired_cells, min_required - 1, 1)
    desired_cells = min(desired_cells, limit)

    cell_width = limit / float(desired_cells)
    search_window = max(
        cell_width * config.walker_search_window_ratio, config.walker_min_search_window
    )
    mean_val = float(np.mean(profile)) if len(profile) else 0.0

    cuts: List[int] = [0]
    for idx in range(1, desired_cells):
        target = cell_width * idx
        prev = cuts[-1]
        if prev + 1 >= limit:
            break

        start = int(math.floor(target - search_window))
        start = max(start, prev + 1, 0)
        end = int(math.ceil(target + search_window))
        end = min(end, limit - 1)
        if end < start:
            start = prev + 1
            end = start

        segment = np.asarray(profile[start : min(end + 1, len(profile))])
        if segment.size:
            best_rel = int(np.argmax(segment))
            best_val = float(segment[best_rel])
            best_idx = start + best_rel
        else:
            best_val = -1.0
            best_idx = start

        strength_threshold = mean_val * config.walker_strength_threshold
        if best_val < strength_threshold:
            fallback_idx = int(round(target))
            if fallback_idx <= prev:
                fallback_idx = prev + 1
            if fallback_idx >= limit:
                fallback_idx = max(limit - 1, prev + 1)
            best_idx = fallback_idx

        cuts.append(best_idx)

    if cuts[-1] != limit:
        cuts.append(limit)

    return sanitize_cuts(cuts, limit)


def stabilize_cuts(
    profile: Sequence[float],
    cuts: List[int],
    limit: int,
    sibling_cuts: Sequence[int],
    sibling_limit: int,
    config: SnapConfig,
) -> List[int]:
    if limit == 0:
        return [0]

    cuts = sanitize_cuts(cuts, limit)
    min_required = max(config.min_cuts_per_axis, 2)
    min_required = min(min_required, limit + 1)

    axis_cells = max(len(cuts) - 1, 0)
    sibling_cells = max(len(sibling_cuts) - 1, 0)
    sibling_has_grid = (
        sibling_limit > 0 and sibling_cells >= (min_required - 1) and sibling_cells > 0
    )
    steps_skewed = False
    if sibling_has_grid and axis_cells > 0:
        axis_step = limit / float(axis_cells)
        sibling_step = sibling_limit / float(sibling_cells)
        step_ratio = axis_step / sibling_step
        steps_skewed = (
            step_ratio > config.max_step_ratio
            or step_ratio < 1.0 / config.max_step_ratio
        )

    has_enough = len(cuts) >= min_required
    if has_enough and not steps_skewed:
        return cuts

    if sibling_has_grid:
        target_step = sibling_limit / float(sibling_cells)
    elif config.fallback_target_segments > 1:
        target_step = limit / float(config.fallback_target_segments)
    elif axis_cells > 0:
        target_step = limit / float(axis_cells)
    else:
        target_step = float(limit)

    if not math.isfinite(target_step) or target_step <= 0.0:
        target_step = 1.0

    return snap_uniform_cuts(profile, limit, target_step, config, min_required)


def stabilize_both_axes(
    profile_x: Sequence[float],
    profile_y: Sequence[float],
    raw_col_cuts: List[int],
    raw_row_cuts: List[int],
    width: int,
    height: int,
    config: SnapConfig,
) -> Tuple[List[int], List[int]]:
    col_cuts_pass1 = stabilize_cuts(
        profile_x, list(raw_col_cuts), width, raw_row_cuts, height, config
    )
    row_cuts_pass1 = stabilize_cuts(
        profile_y, list(raw_row_cuts), height, raw_col_cuts, width, config
    )

    col_cells = max(len(col_cuts_pass1) - 1, 1)
    row_cells = max(len(row_cuts_pass1) - 1, 1)
    col_step = width / float(col_cells)
    row_step = height / float(row_cells)
    step_ratio = col_step / row_step if col_step > row_step else row_step / col_step

    if step_ratio > config.max_step_ratio:
        target_step = min(col_step, row_step)
        if col_step > target_step * 1.2:
            final_cols = snap_uniform_cuts(
                profile_x, width, target_step, config, config.min_cuts_per_axis
            )
        else:
            final_cols = col_cuts_pass1

        if row_step > target_step * 1.2:
            final_rows = snap_uniform_cuts(
                profile_y, height, target_step, config, config.min_cuts_per_axis
            )
        else:
            final_rows = row_cuts_pass1
        return final_cols, final_rows

    return col_cuts_pass1, row_cuts_pass1


def resample(img: np.ndarray, cols: Sequence[int], rows: Sequence[int]) -> np.ndarray:
    if len(cols) < 2 or len(rows) < 2:
        raise PixelSnapperError("Insufficient grid cuts for resampling")

    out_w = max(len(cols) - 1, 1)
    out_h = max(len(rows) - 1, 1)
    final_img = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    for y_i, (ys, ye) in enumerate(zip(rows[:-1], rows[1:])):
        for x_i, (xs, xe) in enumerate(zip(cols[:-1], cols[1:])):
            if xe <= xs or ye <= ys:
                continue
            cell = img[ys:ye, xs:xe]
            if cell.size == 0:
                continue
            pixels = cell.reshape(-1, 3)
            values, counts = np.unique(pixels, axis=0, return_counts=True)
            best_idx = int(np.argmax(counts))
            final_img[y_i, x_i] = values[best_idx]

    return final_img


# ---------------------------------------------------------------------------
# Public entry point: full snap pipeline on one frame
# ---------------------------------------------------------------------------

def apply_pixel_snap(
    img: np.ndarray,
    config: SnapConfig,
    output_scale: int = 1,
) -> np.ndarray:
    """
    Full pixel-snap pipeline on a single (H, W, 3) uint8 frame:
    K-Means quantize → gradient profiles → step estimate → walk →
    stabilize → resample. Optionally upscale the result with nearest-
    neighbour.

    Returns a (out_h*scale, out_w*scale, 3) uint8 image. The output
    dimensions differ from the input because resample collapses each
    detected grid cell into one pixel.
    """
    h, w, _ = img.shape

    # Guard against NaN / Inf inputs (can appear with half-precision pipelines)
    if not np.isfinite(img).all():
        raise PixelSnapperError("Input image contains NaN or Inf values.")

    validate_image_dimensions(w, h)
    if config.pixel_size_override != 0.0:
        max_pixel_size = min(w, h) / 2.0
        px = config.pixel_size_override
        if not math.isfinite(px) or px < 1.0 or px > max_pixel_size:
            raise PixelSnapperError(
                f"pixel_size_override {px:.1f} is out of valid range "
                f"[1, {max_pixel_size:.1f}]"
            )

    quantized = quantize_image(img, config)
    profile_x, profile_y = compute_profiles(quantized)

    step_x_opt = estimate_step_size(profile_x, config)
    step_y_opt = estimate_step_size(profile_y, config)
    step_x, step_y = resolve_step_sizes(step_x_opt, step_y_opt, w, h, config)

    raw_col_cuts = walk(profile_x, step_x, w, config)
    raw_row_cuts = walk(profile_y, step_y, h, config)

    col_cuts, row_cuts = stabilize_both_axes(
        profile_x, profile_y, raw_col_cuts, raw_row_cuts, w, h, config
    )

    snapped = resample(quantized, col_cuts, row_cuts)
    if output_scale > 1:
        snapped = upscale_nearest(snapped, output_scale)
    return snapped
