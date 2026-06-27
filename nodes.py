"""
nodes.py — SpriteSheetExtractor node for ComfyUI

Receives a batch of IMAGE frames (e.g. from WAN 2.2 via any sampler),
samples N evenly-spaced frames, removes the generated background via
per-frame YCbCr chroma keying flood-filled from the canvas borders,
and outputs a horizontal RGBA sprite sheet PNG.

After running, the "Tweak Tolerance" button on the node opens an
interactive panel where the tolerance can be adjusted and re-saved
without re-running the WAN generation.
"""

import os
from collections import deque

import numpy as np
import torch
from PIL import Image

import folder_paths
from .pixel_snapper import apply_pixel_snap, SnapConfig

# ---------------------------------------------------------------------------
# Pillow NEAREST constant — compatible with both old and new Pillow.
# ---------------------------------------------------------------------------
try:
    _NEAREST = Image.Resampling.NEAREST
except AttributeError:
    _NEAREST = Image.NEAREST

# ---------------------------------------------------------------------------
# Global frame cache: keyed by str(unique_id).
# Populated each run so the /sprite_sheet/preview and /sprite_sheet/save
# API routes can re-render at any tolerance without re-running the workflow.
# ---------------------------------------------------------------------------
_frame_cache: dict = {}


# ---------------------------------------------------------------------------
# Background detection
# ---------------------------------------------------------------------------

def detect_background_color(frame_u8: np.ndarray) -> np.ndarray | None:
    """
    Return the dominant background colour as a (3,) uint8 [R, G, B] array.

    Samples only the **top row** and the **top half of the left/right columns**
    because the character is bottom-centre aligned — the bottom row and lower
    sides may contain the character's feet and should not be sampled.

    Chroma (Cb, Cr) is quantised into coarse 16-step bins so the result is
    robust against anti-aliasing and subtle luma gradients in the WAN output.
    """
    H, W = frame_u8.shape[:2]
    half_H = max(H // 2, 1)

    top_row  = frame_u8[0,          :,      :]          # (W,        3)
    left_col = frame_u8[1:half_H,   0,      :]          # (half_H-1, 3)
    rght_col = frame_u8[1:half_H,   W - 1,  :]          # (half_H-1, 3)

    all_px = np.concatenate([top_row, left_col, rght_col], axis=0).astype(np.float32)
    if len(all_px) == 0:
        return None

    R, G, B = all_px[:, 0], all_px[:, 1], all_px[:, 2]
    Cb = np.round(-0.169 * R - 0.331 * G + 0.500 * B + 128).astype(np.int32)
    Cr = np.round( 0.500 * R - 0.419 * G - 0.081 * B + 128).astype(np.int32)

    bin_key = ((Cb >> 4) << 8) | (Cr >> 4)
    unique_keys, counts = np.unique(bin_key, return_counts=True)
    best_key = unique_keys[counts.argmax()]

    mask     = bin_key == best_key
    mean_rgb = all_px[mask].mean(axis=0).clip(0, 255).round().astype(np.uint8)
    return mean_rgb


# ---------------------------------------------------------------------------
# Chroma keying
# ---------------------------------------------------------------------------

def _ycbcr_core_mask(
    frame_u8:        np.ndarray,
    color_rgb:       np.ndarray,
    tolerance_value: float,
) -> np.ndarray:
    """
    Boolean (H, W) mask of pixels within `tolerance_value` of `color_rgb` in
    YCbCr space (chroma dominant, luma at half weight). Shared core-distance
    logic used by both apply_chroma_key and compute_background_mask so the
    two stay numerically consistent.
    """
    data    = frame_u8.astype(np.float32)
    R, G, B = data[:, :, 0], data[:, :, 1], data[:, :, 2]

    cr, cg, cb_ = float(color_rgb[0]), float(color_rgb[1]), float(color_rgb[2])
    cY  =  0.299 * cr + 0.587 * cg + 0.114 * cb_
    cCb = -0.169 * cr - 0.331 * cg + 0.500 * cb_ + 128
    cCr =  0.500 * cr - 0.419 * cg - 0.081 * cb_ + 128

    Y  =  0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.169 * R - 0.331 * G + 0.500 * B + 128
    Cr =  0.500 * R - 0.419 * G - 0.081 * B + 128

    luma_weight = 0.5
    dC   = np.sqrt((Cb - cCb) ** 2 + (Cr - cCr) ** 2)
    dY   = (Y - cY) * luma_weight
    dist = np.sqrt(dC ** 2 + dY ** 2)

    max_chroma = np.sqrt(2.0 * 256.0 ** 2)
    max_dist   = np.sqrt(max_chroma ** 2 + (255.0 * luma_weight) ** 2)
    core_thr   = (tolerance_value / 100.0) * max_dist

    return dist <= core_thr


def _flood_fill_from_borders(core_mask: np.ndarray) -> np.ndarray:
    """
    Flood-fill `core_mask` starting from all four canvas borders.
    Returns a boolean (H, W) "removed" mask: connected-to-border pixels
    that are also within the core mask. Isolated interior pixels of a
    similar colour are NOT included (e.g. a white interior highlight that
    happens to be near the key colour but isn't actually background).
    """
    H, W = core_mask.shape
    removed = np.zeros((H, W), dtype=bool)
    visited = np.zeros((H, W), dtype=bool)
    queue   = deque()

    def _seed(y: int, x: int) -> None:
        if not visited[y, x]:
            visited[y, x] = True
            if core_mask[y, x]:
                removed[y, x] = True
                queue.append((y, x))

    for x in range(W):
        _seed(0,     x)
        _seed(H - 1, x)
    for y in range(1, H - 1):
        _seed(y, 0)
        _seed(y, W - 1)

    while queue:
        y, x = queue.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and not visited[ny, nx]:
                visited[ny, nx] = True
                if core_mask[ny, nx]:
                    removed[ny, nx] = True
                    queue.append((ny, nx))

    return removed


def compute_background_mask(
    frame_u8:        np.ndarray,
    tolerance_value: float,
) -> np.ndarray | None:
    """
    Detect the background colour and return a boolean (H, W) mask of
    border-connected background pixels — the same region apply_chroma_key
    would key out, but as a plain mask with no alpha/fringe/despill work.

    Used by apply_palette_lock to exclude background from frame-0 palette
    extraction. Returns None if no background colour could be detected.
    """
    bg_color = detect_background_color(frame_u8)
    if bg_color is None:
        return None
    core_mask = _ycbcr_core_mask(frame_u8, bg_color, tolerance_value)
    return _flood_fill_from_borders(core_mask)


def apply_chroma_key(
    frame_u8:        np.ndarray,
    color_rgb:       np.ndarray,
    tolerance_value: float,
) -> np.ndarray:
    """
    Remove the background via YCbCr chroma-keyed flood fill + fringe softening.

    Distance is computed in YCbCr space with chroma (Cb, Cr) dominant and luma
    (Y) weighted at 0.5 — shadows and gradients on the same background colour
    therefore key cleanly instead of leaving a coloured halo.

    The flood fill is seeded from **all four borders**, which catches the full
    background border ring; connected background is removed while any isolated
    interior pixels of a similar colour are preserved.  A second fringe pass
    gives partially-transparent alpha to anti-aliased edge pixels.

    Parameters
    ----------
    frame_u8        : (H, W, 3) uint8
    color_rgb       : (3,)      uint8  [R, G, B]  — the key colour
    tolerance_value : float     0-100

    Returns
    -------
    (H, W, 4) uint8  RGBA
    """
    H, W = frame_u8.shape[:2]
    data  = frame_u8.astype(np.float32)

    core_mask = _ycbcr_core_mask(frame_u8, color_rgb, tolerance_value)
    removed   = _flood_fill_from_borders(core_mask)

    # Recompute dist for fringe softening below (core_mask alone isn't enough
    # — fringe needs the continuous distance, not just the thresholded mask).
    R, G, B = data[:, :, 0], data[:, :, 1], data[:, :, 2]
    cr, cg, cb_ = float(color_rgb[0]), float(color_rgb[1]), float(color_rgb[2])
    cY  =  0.299 * cr + 0.587 * cg + 0.114 * cb_
    cCb = -0.169 * cr - 0.331 * cg + 0.500 * cb_ + 128
    cCr =  0.500 * cr - 0.419 * cg - 0.081 * cb_ + 128
    Y   =  0.299 * R + 0.587 * G + 0.114 * B
    Cb  = -0.169 * R - 0.331 * G + 0.500 * B + 128
    Cr  =  0.500 * R - 0.419 * G - 0.081 * B + 128
    luma_weight = 0.5
    dC   = np.sqrt((Cb - cCb) ** 2 + (Cr - cCr) ** 2)
    dY   = (Y - cY) * luma_weight
    dist = np.sqrt(dC ** 2 + dY ** 2)
    max_chroma = np.sqrt(2.0 * 256.0 ** 2)
    max_dist   = np.sqrt(max_chroma ** 2 + (255.0 * luma_weight) ** 2)
    core_thr   = (tolerance_value / 100.0) * max_dist
    fringe_thr = core_thr + max(core_thr * 0.4, 10.0)

    # ---- Alpha channel -------------------------------------------------
    alpha = np.where(removed, 0.0, 255.0)

    # ---- Fringe softening: 8-connected neighbour count via array shifts -
    removed_f  = removed.astype(np.float32)
    nb_removed = np.zeros((H, W), dtype=np.float32)
    nb_total   = np.zeros((H, W), dtype=np.float32)

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            sy0, sy1 = max(0,  dy),      H + min(0,  dy)
            sx0, sx1 = max(0,  dx),      W + min(0,  dx)
            dy0, dy1 = max(0, -dy),      H + min(0, -dy)
            dx0, dx1 = max(0, -dx),      W + min(0, -dx)
            nb_removed[dy0:dy1, dx0:dx1] += removed_f[sy0:sy1, sx0:sx1]
            nb_total  [dy0:dy1, dx0:dx1] += 1.0

    fringe = (~removed) & (nb_removed > 0) & (dist <= fringe_thr)
    if fringe.any():
        closeness        = 1.0 - np.clip(
            (dist - core_thr) / (fringe_thr - core_thr + 1e-6), 0.0, 1.0
        )
        nb_ratio         = nb_removed / np.maximum(nb_total, 1.0)
        fade             = closeness * nb_ratio
        alpha[fringe]    = np.round(alpha[fringe] * (1.0 - fade[fringe]))

    # ---- Spill suppression ------------------------------------------------
    # Anti-aliased fringe pixels are a premultiplied blend of the subject
    # colour and the background colour.  Since we know the background colour
    # and have computed the alpha, we can invert the blend to recover the
    # true subject colour:
    #
    #   pixel_rgb = alpha × subject + (1 - alpha) × background
    #   subject   = (pixel_rgb − (1 − alpha) × background) / alpha
    #
    # This eliminates the coloured halo that bleeds into the subject's
    # edges — the single biggest visual artifact in chroma keying.
    alpha_n     = alpha / 255.0
    despill_msk = (alpha_n > 0.01) & (alpha_n < 0.99)

    if despill_msk.any():
        inv_alpha = np.where(despill_msk, 1.0 / alpha_n, 0.0)
        one_m_a   = 1.0 - alpha_n
        bg_f      = color_rgb.astype(np.float32)
        corrected = data.copy()

        for c in range(3):
            corrected[:, :, c] = np.where(
                despill_msk,
                ((corrected[:, :, c] - one_m_a * bg_f[c]) * inv_alpha).clip(0, 255),
                corrected[:, :, c],
            )

        frame_u8 = corrected.clip(0, 255).astype(np.uint8)

    alpha_u8 = alpha.clip(0, 255).astype(np.uint8)
    return np.dstack([frame_u8, alpha_u8])


# ---------------------------------------------------------------------------
# Sprite sheet composition
# ---------------------------------------------------------------------------

def build_sprite_sheet(
    raw_frames:        list[np.ndarray],
    tolerance:         float,
    remove_background: bool,
    target_size:       int,
) -> np.ndarray:
    """
    Compose a horizontal RGBA sprite sheet from a list of (H, W, 3) uint8
    frames.  Background removal is per-frame so WAN's drift across the
    sequence is handled automatically.

    Returns
    -------
    (target_size, target_size * N, 4) uint8
    """
    n     = len(raw_frames)
    sheet = np.zeros((target_size, target_size * n, 4), dtype=np.uint8)

    for i, frame in enumerate(raw_frames):
        if remove_background:
            bg_color = detect_background_color(frame)
            if bg_color is not None:
                rgba = apply_chroma_key(frame, bg_color, tolerance)
            else:
                alpha = np.full((target_size, target_size), 255, dtype=np.uint8)
                rgba  = np.dstack([frame, alpha])
        else:
            alpha = np.full((target_size, target_size), 255, dtype=np.uint8)
            rgba  = np.dstack([frame, alpha])

        sheet[:, i * target_size:(i + 1) * target_size, :] = rgba

    return sheet


# ---------------------------------------------------------------------------
# Palette lock — snap frames 1..n to frame 0's colour palette
# ---------------------------------------------------------------------------

def apply_palette_lock(
    raw_frames:       list,
    n_colors:         int,
    n_buffer:         int,
    buffer_threshold: float,
    bg_mask_0:        np.ndarray | None = None,
    bg_masks_rest:    list | None = None,
) -> list:
    """
    Fix WAN temporal colour drift by mapping every pixel in frames 1..n to
    the nearest colour in frame 0's extracted palette.

    A small buffer palette slot is built from colours that appear in later
    frames but are far enough from the frame-0 palette to be genuine transient
    effects (fireballs, hit flashes, etc.) rather than drift.  Only those
    transient colours are left uncorrected.

    Frame 0 is always returned pixel-perfect.

    Parameters
    ----------
    raw_frames       : list of (H, W, 3) uint8 — already cropped + resized
    n_colors         : number of colours to extract from frame 0 (e.g. 32)
    n_buffer         : extra buffer slots for transient colours
    buffer_threshold : RGB Euclidean distance above which a colour is
                       treated as a transient effect, not drift (e.g. 30)
    bg_mask_0        : optional (H, W) bool — True where frame 0 is
                       background.  When given, those pixels are excluded
                       from the frame-0 palette extraction so the limited
                       colour budget isn't spent representing background
                       that's going to be keyed out anyway.
    bg_masks_rest    : optional list of (H, W) bool, one per frame in
                       raw_frames[1:], True where that frame is background.
                       CRITICAL for the transient/buffer-colour scan below:
                       without this, background pixels in frames 1..n are
                       (correctly) far from the now bg-free frame-0 palette,
                       so they get misclassified as "transient effects" and
                       consume the buffer palette slots almost entirely —
                       leaving the combined palette polluted with smeared
                       background-adjacent colours that genuine character
                       pixels can get nearest-neighbour-matched against
                       instead of the correct palette entry. This is the
                       single biggest cause of colour muting after lock,
                       and it gets worse as palette_colors goes up, since
                       a denser legitimate palette doesn't fix a corrupted
                       buffer palette being checked in the same argmin.
    """
    if len(raw_frames) < 2:
        return raw_frames

    # ---- Extract frame-0 palette via median-cut, character pixels only ---
    n_colors = max(1, min(n_colors, 256))

    frame0 = raw_frames[0]
    if bg_mask_0 is not None and bg_mask_0.any() and not bg_mask_0.all():
        fg_px = frame0[~bg_mask_0]                       # (M, 3) character-only pixels
    else:
        fg_px = frame0.reshape(-1, 3)

    # quantize() needs a 2D-ish image; reshape the foreground pixel list
    # into a tall Nx1 strip so it quantizes purely on colour, not layout.
    n_colors_eff = max(1, min(n_colors, len(fg_px)))
    q0 = Image.fromarray(
        fg_px.reshape(-1, 1, 3), "RGB"
    ).quantize(colors=n_colors_eff, method=0)

    flat = q0.getpalette()
    n_actual = len(flat) // 3                              # Pillow may return
    palette_0 = np.array(                                  # fewer than requested
        flat[: n_actual * 3], dtype=np.float32
    ).reshape(n_actual, 3)                                  # (K, 3)
    combined  = palette_0.copy()

    # ---- Collect transient pixels from frames 1..n -----------------------
    if n_buffer > 0 and buffer_threshold > 0:
        far_list = []
        for i, frame in enumerate(raw_frames[1:]):
            px = frame.reshape(-1, 3).astype(np.float32)

            if bg_masks_rest is not None and i < len(bg_masks_rest) and bg_masks_rest[i] is not None:
                fg_only = ~bg_masks_rest[i].reshape(-1)
                px = px[fg_only]                            # drop background pixels —
                if len(px) == 0:                            # they're trivially "far" from
                    continue                                 # the bg-free frame-0 palette
                                                              # and would falsely look transient

            diff = px[:, np.newaxis, :] - palette_0[np.newaxis, :, :]  # (N, K, 3)
            dmin = np.sqrt((diff ** 2).sum(axis=2).min(axis=1))         # (N,)
            far  = px[dmin > buffer_threshold]
            if len(far):
                far_list.append(far)

        if far_list:
            all_far = np.concatenate(far_list).clip(0, 255).astype(np.uint8)
            n_buf   = min(n_buffer, len(all_far), 256)
            if n_buf > 0:
                try:
                    pil_far = Image.fromarray(
                        all_far.reshape(-1, 1, 3), "RGB"
                    )
                    qbuf     = pil_far.quantize(colors=n_buf, method=0)
                    buf_flat = qbuf.getpalette()
                    n_buf_actual = len(buf_flat) // 3
                    buf_pal  = np.array(
                        buf_flat[: n_buf_actual * 3], dtype=np.float32
                    ).reshape(n_buf_actual, 3)
                    combined = np.vstack([palette_0, buf_pal])          # (K+B, 3)
                except Exception as exc:
                    print(f"[SpriteSheetExtractor] Buffer palette error: {exc}")

    # ---- Snap each of frames 1..n to the combined palette ----------------
    # Process in chunks to keep peak memory bounded regardless of frame size.
    CHUNK  = 8192
    result = [raw_frames[0]]    # frame 0 is used as-is

    for frame in raw_frames[1:]:
        H, W, _  = frame.shape
        px       = frame.reshape(-1, 3).astype(np.float32)
        snapped  = np.empty_like(px)

        for s in range(0, len(px), CHUNK):
            chunk   = px[s : s + CHUNK]
            diff    = chunk[:, np.newaxis, :] - combined[np.newaxis, :, :]
            nearest = (diff ** 2).sum(axis=2).argmin(axis=1)
            snapped[s : s + CHUNK] = combined[nearest]

        result.append(snapped.clip(0, 255).astype(np.uint8).reshape(H, W, 3))

    return result


# ---------------------------------------------------------------------------
# ComfyUI node
# ---------------------------------------------------------------------------

class SpriteSheetExtractor:
    """
    Sprite Sheet Extractor
    ----------------------
    Receives IMAGE frames from WAN (or any source), samples N evenly-spaced
    frames, removes the generated background with per-frame YCbCr chroma
    keying, and saves a horizontal PNG sprite sheet to the output directory.

    After the workflow runs, click **Tweak Tolerance** on the node to open an
    interactive panel: adjust the tolerance slider, see the result live, then
    click Save to overwrite the PNG — no need to re-run WAN generation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "target_size": ("INT", {
                    "default": 96, "min": 16, "max": 512, "step": 8,
                    "display": "number",
                    "tooltip": "Output sprite size in pixels (square). "
                               "96 = 8x downscale from 768px WAN output.",
                }),
                "frame_count": ("INT", {
                    "default": 6, "min": 1, "max": 64,
                    "display": "number",
                    "tooltip": "Number of frames to sample from the input batch. "
                               "Frames are picked at even intervals so you can "
                               "generate more than needed and trim here.",
                }),
                "tolerance": ("FLOAT", {
                    "default": 15.0, "min": 0.0, "max": 100.0, "step": 0.5,
                    "display": "slider",
                    "tooltip": "YCbCr chroma-key tolerance (0–100). "
                               "Use Tweak Tolerance after running to adjust without "
                               "re-generating.",
                }),
                "remove_background": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Auto-detect and remove the background per frame.",
                }),
                "palette_lock": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Snap frames 1..n to frame 0's colour palette to fix "
                               "WAN temporal colour drift.  Transient colours (fireballs, "
                               "hit effects) are preserved via the buffer slots.",
                }),
                "palette_colors": ("INT", {
                    "default": 32, "min": 2, "max": 256, "step": 2,
                    "display": "number",
                    "tooltip": "Colours extracted from frame 0.  More = finer detail "
                               "preserved; fewer = stronger colour unification.",
                }),
                "buffer_colors": ("INT", {
                    "default": 8, "min": 0, "max": 64, "step": 1,
                    "display": "number",
                    "tooltip": "Extra palette slots for transient colours not present in "
                               "frame 0 (fireballs, flashes, etc.).",
                }),
                "buffer_threshold": ("FLOAT", {
                    "default": 30.0, "min": 0.0, "max": 100.0, "step": 1.0,
                    "display": "slider",
                    "tooltip": "RGB distance above which a colour is considered a "
                               "transient effect rather than drift.  25–35 suits most "
                               "pixel art; lower = more aggressive snapping.",
                }),
                "filename_prefix": ("STRING", {
                    "default": "sprite_sheet",
                    "tooltip": "Output filename prefix.  Each run auto-increments the "
                               "counter:  sprite_sheet_00001_.png, _00002_.png …",
                }),
                "pixel_snap": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Pixelate each frame with the grid-snapping pipeline after "
                               "downscaling.  The snapped image is always resized back to "
                               "target_size so the canvas and subject-to-frame ratio are "
                               "preserved regardless of snap_pixel_size.",
                }),
                "snap_pixel_size": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 32.0, "step": 1.0,
                    "display": "number",
                    "tooltip": "Pixel-art cell size in source pixels.  "
                               "0.0 = auto-detect the natural grid from gradient peaks.  "
                               "Valid override values start at 1.0.  Non-integer values "
                               "(e.g. 7.0, 6.5) are fully supported — the output is always "
                               "nearest-neighbour upscaled back to target_size so the "
                               "subject ratio is identical to the input.",
                }),
                "snap_colors": ("INT", {
                    "default": 16, "min": 2, "max": 64, "step": 2,
                    "display": "number",
                    "tooltip": "K-Means palette size used during pixel snapping.  "
                               "The frame is quantised to this many colours before the "
                               "grid is detected — fewer colours simplify the image more "
                               "aggressively and give cleaner block edges; more colours "
                               "preserve finer detail before snapping.  "
                               "8 = maximum simplification, 32 = fine detail.  "
                               "Has no effect when pixel_snap is off.",
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES  = ("IMAGE", "MASK")
    RETURN_NAMES  = ("sprite_sheet", "alpha")
    FUNCTION      = "extract"
    CATEGORY      = "image/animation"
    OUTPUT_NODE   = True

    def extract(
        self,
        images:            torch.Tensor,
        target_size:       int,
        frame_count:       int,
        tolerance:         float,
        remove_background: bool,
        palette_lock:      bool,
        palette_colors:    int,
        buffer_colors:     int,
        buffer_threshold:  float,
        filename_prefix:   str,
        pixel_snap:        bool,
        snap_pixel_size:   float,
        snap_colors:       int,
        unique_id:         str,
    ):
        """
        images : (B, H, W, C) float32  0-1   ComfyUI IMAGE batch
        """
        B     = images.shape[0]
        count = max(1, min(frame_count, B))

        # Even temporal sample — mirrors the browser tool's index formula.
        indices = (
            [0]
            if count == 1
            else [min(round(B * i / count), B - 1) for i in range(count)]
        )

        # Build SnapConfig once — config is the same for every frame.
        # snap_pixel_size is clamped against the CROPPED source size (not
        # target_size) because pixel snap runs before the PIL downscale.
        # For WAN 2.2 at 768 px with snap_pixel_size=8: 768÷8 = 96 cells →
        # the snap output already equals target_size with no resize needed.
        if pixel_snap:
            input_side = min(images.shape[1], images.shape[2])
            _eff_size = snap_pixel_size
            if _eff_size != 0.0:
                _max_override = input_side / 2.0
                if _eff_size > _max_override:
                    print(
                        f"[SpriteSheetExtractor] snap_pixel_size {_eff_size} clamped to "
                        f"{_max_override} (input_side/2)"
                    )
                    _eff_size = _max_override
            snap_cfg = SnapConfig(pixel_size_override=_eff_size, k_colors=snap_colors)

        # Decode → center-crop to square → pixel snap on full-res (if enabled)
        # → nearest-neighbour downscale to target_size.
        # Snapping at full resolution is essential: at target_size (e.g. 96 px),
        # snap_pixel_size=8 would give only 12×12 art pixels, which looks
        # extremely blocky.  On the full-res source (e.g. 768 px), the same
        # snap_pixel_size=8 gives 96×96 art pixels — exactly target_size.
        raw_frames: list[np.ndarray] = []
        for idx in indices:
            frame_f  = images[idx].cpu().numpy()                        # (H, W, C) 0-1
            frame_u8 = (frame_f * 255).clip(0, 255).astype(np.uint8)   # (H, W, C)
            H, W     = frame_u8.shape[:2]
            side     = min(H, W)
            sy, sx   = (H - side) // 2, (W - side) // 2
            cropped  = frame_u8[sy:sy + side, sx:sx + side, :3]        # full-res square

            if pixel_snap:
                try:
                    snapped = apply_pixel_snap(cropped, snap_cfg)
                    # Resize snap output to target_size.  When snap_pixel_size
                    # divides the source side exactly (e.g. 768÷8=96) the snap
                    # output is already target_size and this is a no-op.
                    frame_np = np.array(
                        Image.fromarray(snapped).resize(
                            (target_size, target_size), _NEAREST
                        )
                    )
                except Exception as exc:
                    print(
                        f"[SpriteSheetExtractor] pixel_snap failed for frame {idx}: {exc}"
                    )
                    frame_np = np.array(
                        Image.fromarray(cropped).resize(
                            (target_size, target_size), _NEAREST
                        )
                    )
            else:
                frame_np = np.array(
                    Image.fromarray(cropped).resize((target_size, target_size), _NEAREST)
                )
            raw_frames.append(frame_np)

        # Optional: palette-lock to fix temporal colour drift.
        # Runs after pixel snap (on target_size frames) — the two stages no
        # longer conflict because snap operated at full resolution and lock
        # operates at target_size.
        if palette_lock and len(raw_frames) > 1:
            if remove_background:
                bg_mask_0     = compute_background_mask(raw_frames[0], tolerance)
                bg_masks_rest = [
                    compute_background_mask(f, tolerance) for f in raw_frames[1:]
                ]
            else:
                bg_mask_0     = None
                bg_masks_rest = None
            raw_frames = apply_palette_lock(
                raw_frames, palette_colors, buffer_colors, buffer_threshold,
                bg_mask_0=bg_mask_0, bg_masks_rest=bg_masks_rest,
            )

        # Cache raw (possibly locked) frames so the interactive API can re-render
        _frame_cache[str(unique_id)] = {
            "frames":          raw_frames,
            "size":            target_size,
            "remove_bg":       remove_background,
            "filename_prefix": filename_prefix,
        }

        # Build sprite sheet
        sheet_rgba = build_sprite_sheet(raw_frames, tolerance, remove_background, target_size)

        # Auto-incrementing save path — same convention as ComfyUI's SaveImage
        # so the filename never silently overwrites a previous run.
        # e.g. sprite_sheet_00001_.png, sprite_sheet_00002_.png …
        output_dir = folder_paths.get_output_directory()
        full_dir, base, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, output_dir, sheet_rgba.shape[1], sheet_rgba.shape[0]
        )
        filename = f"{base}_{counter:05d}_.png"
        out_path = os.path.join(full_dir, filename)

        Image.fromarray(sheet_rgba, "RGBA").save(out_path)
        print(f"[SpriteSheetExtractor] Saved: {out_path}")

        # Cache the actual saved path so the interactive panel can overwrite
        # this specific file (not generate a new incremented one).
        _frame_cache[str(unique_id)]["saved_path"] = out_path
        _frame_cache[str(unique_id)]["filename"]   = filename

        # Return IMAGE (RGB) + MASK (alpha) for downstream ComfyUI use
        sheet_rgb_f   = sheet_rgba[:, :, :3].astype(np.float32) / 255.0
        sheet_alpha_f = sheet_rgba[:, :,  3].astype(np.float32) / 255.0

        sheet_t = torch.from_numpy(sheet_rgb_f  ).unsqueeze(0)  # (1, H, W*N, 3)
        alpha_t = torch.from_numpy(sheet_alpha_f).unsqueeze(0)  # (1, H, W*N)

        return (sheet_t, alpha_t)


# ---------------------------------------------------------------------------
# Animated preview node
# ---------------------------------------------------------------------------

class SpriteSheetPreview:
    """
    Sprite Sheet Preview (animated)
    --------------------------------
    Connects to the sprite_sheet AND alpha outputs of SpriteSheetExtractor.
    The node chops the horizontal sheet back into individual frames
    (sheet_width ÷ target_size) and displays them looping as a pixel-art
    animation directly inside the node at a configurable FPS.

    Wire the alpha MASK output from the extractor into the optional alpha
    input here to see the correctly-keyed result on a checkerboard — without
    it the preview shows the RGB with black where the background was removed.

    target_size is set separately so this node can stand alone or sit next
    to the extractor without needing the value wired across.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sprite_sheet": ("IMAGE",),
                "target_size": ("INT", {
                    "default": 96, "min": 16, "max": 512, "step": 8,
                    "display": "number",
                    "tooltip": "Must match the target_size used when the sheet was built.",
                }),
                "fps": ("INT", {
                    "default": 6, "min": 1, "max": 60,
                    "display": "number",
                    "tooltip": "Playback speed of the animated preview in the node.",
                }),
            },
            "optional": {
                "alpha": ("MASK", {
                    "tooltip": "Connect the alpha output from Sprite Sheet Extractor to "
                               "see background removal on the checkerboard preview.",
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ()
    FUNCTION     = "preview"
    CATEGORY     = "image/animation"
    OUTPUT_NODE  = True

    def preview(self, sprite_sheet, target_size, fps, unique_id, alpha=None):
        # sprite_sheet : (B, H, W, 3) float32 0-1  (RGB from extractor)
        # alpha        : (B, H, W)    float32 0-1  (MASK from extractor, optional)
        img     = sprite_sheet[0]           # (H, W, 3)
        H, W, C = img.shape
        frame_count = max(1, W // target_size)

        rgb_u8 = (img.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)[:, :, :3]

        if alpha is not None:
            # Recombine RGB + alpha to get the background-removed RGBA result
            alpha_u8 = (alpha[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)  # (H, W)
            arr = np.dstack([rgb_u8, alpha_u8])   # (H, W, 4)
            pil = Image.fromarray(arr, "RGBA")
        else:
            pil = Image.fromarray(rgb_u8, "RGB")

        # Save to temp — one file per node instance to avoid collisions
        temp_dir = folder_paths.get_temp_directory()
        filename = f"ss_anim_{unique_id}.png"
        pil.save(os.path.join(temp_dir, filename))

        return {
            "ui": {
                # ComfyUI picks this up and makes the image accessible via /view
                "images": [{"filename": filename, "subfolder": "", "type": "temp"}],
                # Extra metadata consumed by the JS extension
                "sprite_meta": [{"target_size": target_size, "fps": fps, "frame_count": frame_count}],
            }
        }