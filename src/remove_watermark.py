"""remove_watermark.py -- general-purpose batch watermark removal.

Design goal: work on watermarks this script has never seen -- any position, any
colour, light or dark, opaque or semi-transparent, over any background.

What a watermark actually is
----------------------------
Every mark this tool handles is an *overlay*: over the marked pixels the image
satisfies

    observed = (1 - a) * original + a * color          (a in [0,1], per pixel)

with `a` and `color` fixed by the mark. That single sentence is the whole tool:

  * it says the mark's contribution is a **residual vector** r = observed - original
    which points in ONE direction over the whole mark (toward `color`), whatever
    the mark's colour, and whatever the picture underneath;
  * it says texture has no such direction: its residuals cancel, so the mean of r
    over a texture blob is ~0 and the pixels disagree with each other;
  * and when `a` and `color` are known it is invertible exactly:
    original = (observed - a * color) / (1 - a).

So detection is: estimate the local background, take the residual vector field,
and keep the components whose residuals agree in direction and are strong enough.
Restoration is the inverted formula -- not a blur patch.

Strategies
----------
  single    (default) one image, residual-vector detector above.
  multi     N images sharing one mark: per-pixel spread collapses where the mark
            sits. Needs >= 3 same-size images.
  template  you supply the mark (PNG with alpha, or black-on-white shape). The
            shape becomes the mask; with --restore it is inverted exactly.
  --learn   you supply pairs (marked + clean) of the same picture: the tool solves
            `a` and `color` from the pair, writes a reusable signature PNG, and
            reports how well the model explains the pair.
  --period  tiled mark: fold the residual modulo the lattice period (auto or
            given) so the repeated mark reinforces itself and the picture averages
            out, then replicate the recovered shape over the lattice.

Usage
-----
    python src/remove_watermark.py <input...> [options]

<input> is a file or a directory (directories are walked). Outputs default to
`<dir>/clean/<name>-clean.png`; ORIGINALS ARE NEVER MODIFIED.

    # point it at a folder and let it decide (recommended first try)
    python src/remove_watermark.py incoming/

    # mark in a known area only (faster, safer on busy frames)
    python src/remove_watermark.py incoming/ --search bottom-right

    # you know the box exactly (x,y,w,h; repeatable)
    python src/remove_watermark.py incoming/ --rect 1980,1560,320,160 --search none

    # you have the mark itself: exact inversion (needs alpha, or --alpha/--color)
    python src/remove_watermark.py incoming/ --template mark.png --restore

    # teach it the mark once from a marked/clean pair, reuse forever after
    python src/remove_watermark.py --learn shot_wm.png shot_clean.png \
        --signature-out jianying.png
    python src/remove_watermark.py incoming/ --template jianying.png --restore

    # repeated/tiled mark (period found automatically unless you pass one)
    python src/remove_watermark.py incoming/ --period auto

    # a whole batch shot with the SAME mark in the SAME place (strongest)
    python src/remove_watermark.py batch/ --strategy multi

    # see what it would change, write nothing; dump the mask to look at
    python src/remove_watermark.py incoming/ --dry-run --mask-out-dir masks/

Exit code: 0 = all ok, 1 = some file failed, 2 = bad arguments.
"""
import argparse
import json
import os
import sys

try:
    import cv2
    import numpy as np
except ImportError as exc:  # pragma: no cover
    print("[FAIL] needs opencv-python and numpy: %s" % exc)
    sys.exit(2)

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")

SIGNATURE_SCHEMA = "dsh-watermark/signature@1"
REPORT_SCHEMA = "dsh-watermark/report@1"


# --------------------------------------------------------------------------
# I/O helpers -- MUST go through imencode/fromfile, never cv2.imread/imwrite.
# Measured bug (this repo lives at D:\<chinese chars>\): cv2.imwrite returns
# False on a non-ASCII path, so every output silently failed to appear and the
# tool looked like it "found no images". Same trap on the read side.
# --------------------------------------------------------------------------
def imread_any(path, flags=cv2.IMREAD_UNCHANGED):
    try:
        buf = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, flags)


def imwrite_any(path, img):
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    try:
        buf.tofile(path)
    except OSError:
        return False
    return True


# Named search windows, as fractions of the frame (x0, x1, y0, y1).
SEARCH = {
    "all":          (0.00, 1.00, 0.00, 1.00),
    "bottom-right": (0.50, 1.00, 0.75, 1.00),
    "bottom-left":  (0.00, 0.50, 0.75, 1.00),
    "top-right":    (0.50, 1.00, 0.00, 0.25),
    "top-left":     (0.00, 0.50, 0.00, 0.25),
    "bottom":       (0.00, 1.00, 0.80, 1.00),
    "top":          (0.00, 1.00, 0.00, 0.20),
    "corners":      (0.00, 1.00, 0.00, 1.00),   # all four corner quartiles
    "none":         (0.00, 0.00, 0.00, 0.00),
}

CORNER_BOXES = [(0.00, 0.35, 0.00, 0.20), (0.65, 1.00, 0.00, 0.20),
                (0.00, 0.35, 0.80, 1.00), (0.65, 1.00, 0.80, 1.00)]


def parse_period(value):
    """'W,H' -> (W, H). Separate from parse_pairs because a period is 2 numbers."""
    p = value.split(",")
    if len(p) != 2:
        raise ValueError("--period needs W,H (got %r)" % value)
    return int(float(p[0])), int(float(p[1]))


def parse_pairs(values):
    out = []
    for v in values or []:
        p = v.split(",")
        if len(p) != 4:
            raise ValueError("need x,y,w,h (got %r)" % v)
        out.append(tuple(int(float(x)) for x in p))
    return out


def parse_color(value):
    p = [x for x in value.replace(" ", "").split(",") if x]
    if len(p) != 3:
        raise ValueError("--color needs B,G,R (got %r)" % value)
    return np.array([float(x) for x in p], np.float32)


def search_mask(shape, search, rects):
    """Mask of where we are ALLOWED to look (255 = inside the search area)."""
    h, w = shape[:2]
    allow = np.zeros((h, w), np.uint8)
    if search != "none":
        boxes = CORNER_BOXES if search == "corners" else [SEARCH[search]]
        for fx0, fx1, fy0, fy1 in boxes:
            allow[int(fy0 * h):int(fy1 * h), int(fx0 * w):int(fx1 * w)] = 255
    for (x, y, rw, rh) in rects:
        allow[max(0, y):y + rh, max(0, x):x + rw] = 255
    return allow


# --------------------------------------------------------------------------
# Background estimation
# --------------------------------------------------------------------------
def _odd(k):
    k = int(k)
    return k if k % 2 == 1 else k + 1


def median_bg(plane, k, max_k=31):
    """Local background of one 8-bit plane: a MEDIAN filter, not a mean.

    Measured why median (this is the fix for the "big mark collapses" case in
    README §3): a mean filter's window is contaminated by the mark's own pixels
    whenever the window is not much larger than the mark, so `plane - mean`
    shrinks toward zero exactly where the mark is -- the mark cancels itself out.
    A median returns the majority value in the window, so as long as the mark
    covers less than half of it the estimate stays on the background and the
    residual keeps the mark's full amplitude.

    Large kernels are computed on a downscaled copy: the result is a background
    estimate, so the extra smoothing is harmless and the cost stays flat.
    """
    k = _odd(max(3, k))
    if k <= max_k:
        return cv2.medianBlur(plane, k)
    h, w = plane.shape[:2]
    f = int(np.ceil(k / float(max_k)))
    small = cv2.resize(plane, (max(3, w // f), max(3, h // f)), interpolation=cv2.INTER_AREA)
    small = cv2.medianBlur(small, _odd(max(3, k // f)))
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def residuals_for_radius(bgr, radius):
    """Residual vector field r = pixel - local background, all 3 channels.

    Keeping the full vector (not a grey-scalar difference) is what makes a
    coloured mark visible on a background of the SAME luminance: a magenta
    "DEMO" on a magenta-luma gradient has r ~ ((+150,-150,+150)) -- invisible in
    grey, unmissable as a vector.
    """
    k = 2 * radius + 1
    out = np.empty(bgr.shape, np.float32)
    for c in range(3):
        out[:, :, c] = bgr[:, :, c].astype(np.float32) - median_bg(bgr[:, :, c], k)
    return out


def vector_norm(res):
    return np.sqrt(np.einsum("ijk,ijk->ij", res, res))


LAPLACIAN_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)


def laplacian_magnitude(bgr):
    """High-frequency energy of every pixel, as the norm over the 3 channels.

    Used both to estimate the frame's noise (the median of it) and to test for the
    texture collapse an overlay causes (the local value of it). For iid noise of
    std s the kernel has ||L||^2 = 36, so the per-channel response has std 6s.
    """
    acc = None
    for c in range(3):
        plane = bgr[:, :, c].astype(np.float32)
        lap = cv2.filter2D(plane, -1, LAPLACIAN_KERNEL, borderType=cv2.BORDER_REPLICATE)
        acc = lap * lap if acc is None else acc + lap * lap
    return np.sqrt(acc)


def estimate_noise_sigma(bgr, energy=None):
    """Robust additive-noise sigma of the frame, as a 3-vector norm (per channel,
    then combined).

    Two measured requirements, both from the first run of the new detector:

      * it must be GLOBAL. The first implementation used `blur(|residual|)` as a
        local noise scale -- i.e. the mark's own edges fed the floor that was
        supposed to detect them. On every synthetic case that floor rose to 3-4x
        the mark's residual and the contrast vote switched off completely (0
        pixels): the detector disabled itself exactly where the signal was.
      * it must not count PICTURE STRUCTURE as noise. A median-filter difference
        returns ~0 on smooth content but still follows large blurred blobs, which
        on the busy synthetic background pushed sigma to 23-27 (true noise: 12.5)
        and the floor to ~70, burying a 35%-alpha mark.

    So: the Immerkaer Laplacian estimator, with the median of |Laplacian| instead
    of the mean so a few strong edges cannot inflate it, and per channel before
    combining: std ~= 1.4826 * median|lap| / 6.
    """
    if energy is None:
        energy = laplacian_magnitude(bgr)
    per_channel = 1.4826 * float(np.median(energy)) / 6.0
    return float(np.sqrt(3.0) * per_channel)


def scale_radii(shape, args):
    """Background window sizes used to build the residual field.

    A single window cannot work at both ends (measured): smaller than the mark's
    stroke -> the "background" is the mark; much larger than a thin stroke -> the
    stroke is averaged away. So a small pyramid {window/4, window/2, window}.

    Deliberately NOT included by default: windows at the scale of the whole frame.
    Those were tried and made things worse -- on a picture made of large soft
    blobs a frame-scale median turns every blob into a "residual", so 50-67% of
    the frame became seed pixels and the components merged into one blob that no
    filter can judge. A mark is found by its boundary; the interior follows from
    filling that boundary, so the fine pyramid is enough. --auto-scales adds the
    image-relative scales back for callers who want to try them.
    """
    h, w = shape[:2]
    m = min(h, w)
    radii = [max(2, args.window // 4), max(3, args.window // 2), args.window]
    if args.auto_scales:
        radii += [max(args.window, m // 8)]
    return sorted({int(r) for r in radii if r >= 2})


def residual_field(bgr, args):
    """Multi-scale residual vectors + per-scale vote count.

    Returns (best (H,W,3) float32, best_mag (H,W), votes (H,W) uint8, floor).
    `best` is the residual at the scale where this pixel's response is strongest
    -- not a sum, so a thin stroke and a fat blob are both kept at full strength.
    """
    floor = max(args.contrast_delta, args.k_sigma * estimate_noise_sigma(bgr))
    best = best_mag = votes = None
    for r in scale_radii(bgr.shape, args):
        res = residuals_for_radius(bgr, r)
        mag = vector_norm(res)
        if best is None:
            best, best_mag = res, mag
            votes = (mag > floor).astype(np.uint8)
        else:
            take = mag > best_mag
            best[take] = res[take]
            best_mag[take] = mag[take]
            votes += (mag > floor).astype(np.uint8)
            del res, mag
    return best, best_mag, votes, floor


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------
def _fill_holes(binary):
    """Fill the enclosed holes of a boolean mask (not its convex hull)."""
    h, w = binary.shape
    big = np.zeros((h + 2, w + 2), np.uint8)
    big[1:-1, 1:-1] = binary.astype(np.uint8)
    ff = 1 - big
    cv2.floodFill(ff, None, (0, 0), 0)
    return (big | (ff > 0))[1:-1, 1:-1].astype(bool)


def _ring_inner(inner, field, bbox, kernel_r, allow, erode_r=0):
    """Median of `field` inside `inner` vs. in the ring around it.

    Returns (innerLevel, ringLevel). Both are medians, so a few strong pixels (a
    glyph's antialiased edge) cannot move either number.
    """
    x, y, w, h = bbox
    pad = kernel_r + 2
    y0, x0 = max(0, y - pad), max(0, x - pad)
    y1 = min(field.shape[0], y + h + pad)
    x1 = min(field.shape[1], x + w + pad)
    sub = inner[y0:y1, x0:x1].astype(np.uint8)
    sub_field = field[y0:y1, x0:x1]
    sub_allow = allow[y0:y1, x0:x1] > 0
    if erode_r > 0:
        sub_core = cv2.erode(sub, np.ones((2 * erode_r + 1, 2 * erode_r + 1), np.uint8)) > 0
        if not sub_core.any():
            # Erosion emptied a small component. Falling back to "no interior" would
            # report level 0, which reads as a PERFECT texture collapse and let every
            # small false positive through (measured: 7 spurious components on the
            # clean frame, each 40-100 px). Measure the whole component instead.
            sub_core = sub > 0
    else:
        sub_core = sub > 0
    ring = (cv2.dilate(sub, np.ones((2 * kernel_r + 1, 2 * kernel_r + 1), np.uint8)) > 0)
    ring &= (sub == 0) & sub_allow
    inner_level = float(np.median(sub_field[sub_core])) if sub_core.any() else 0.0
    ring_level = float(np.median(sub_field[ring])) if ring.any() else 0.0
    return inner_level, ring_level


def detect_single(bgr, allow, args, shape_mask=None):
    """Residual-vector detector: locally-normalised seeds + component tests.

    Five decisions, in order:
      1. LOCAL LEVEL  the residual level of each pixel's own neighbourhood, as a
                median over a window several times the base scale. A thin stroke
                does not move its own window's median, so this measures the
                PICTURE's residual level at that point, not the mark's.
      2. SEED   pixels whose residual both clears the frame noise floor and beats
                their own neighbourhood by --saliency-ratio. This is what makes a
                busy background survivable: measured on the clean busy frame the
                artwork's residual is 40-90 with its own local level tracking it
                (ratio ~1), while a mark's stroke reads 120-290 over a local level
                of ~40 (ratio 3-6).
      3. GROW   hysteresis, also relative to the local level, so growth cannot
                leak into artwork whose residual is merely high in absolute terms.
                This is how the flat interior of a glyph gets in: the interior of
                an opaque mark has no residual of its own.
      4. JUDGE  per component: area, residual strength, and DIRECTION AGREEMENT --
                the share of the component's pixels whose residual vector points
                the same way as the component's mean. A mark sits on one side of
                the picture over its whole extent; texture flips direction.
      5. SALIENCY  the component as a whole must still stand out from the ring
                around it, which rejects a blob of artwork that merged with
                something sharp.
    """
    best, best_mag, votes, floor = residual_field(bgr, args)

    # The frame's high-frequency energy, and the level below which a neighbourhood
    # counts as "flat" (nothing for an overlay to flatten). Self-calibrating: half
    # the frame's own median energy.
    energy = laplacian_magnitude(bgr)
    median_e = float(np.median(energy[allow > 0])) if (allow > 0).any() else 0.0
    flat_knee = max(args.flat_energy, args.flat_energy_frac * median_e)

    # 1. the picture's own residual level, locally (robust to a thin stroke)
    k = _odd(2 * args.window + 1)
    level = cv2.medianBlur(np.clip(best_mag, 0, 255).astype(np.uint8), k).astype(np.float32)

    # 2. seeds: stronger than the frame floor AND stronger than the neighbourhood
    res_seed = (best_mag > floor) & (votes >= max(1, args.min_votes))
    if args.saliency_ratio > 0:
        res_seed &= best_mag > args.saliency_ratio * level

    # 2b. MATCHED-FILTER seeds: the mean residual VECTOR over a small window.
    #
    #     Per-pixel thresholds are limited by the frame's noise: a 35%-alpha mark
    #     over a noisy frame moves a single pixel's residual by only ~2.5 sigma,
    #     which no threshold can separate from the noise (measured: that case
    #     seeded nothing and was missed outright, hit 0/1175). But noise is
    #     zero-mean and directionless, so averaging the residual VECTORS over a
    #     window leaves the mark's coherent residual almost untouched while the
    #     noise falls by sqrt(N) -- the standard matched filter. Over a 5x5 window
    #     the noise floor drops 5x, which is exactly what a 2.5-sigma mark needs.
    #     The window is deliberately small: it must stay near the stroke width, or
    #     a thin stroke is averaged away with the noise.
    mw = _odd(max(3, args.match_window))
    if args.match_window > 0:
        mag_mean = vector_norm(cv2.blur(best, (mw, mw)))
        mean_floor = max(args.contrast_delta, args.k_sigma * estimate_noise_sigma(bgr, energy)) \
            / float(mw)
        seed_mean = mag_mean > mean_floor * args.match_gain
    else:
        seed_mean = np.zeros(bgr.shape[:2], bool)

    # 2c. texture-collapse seeds -- the aggregate cue, used where the per-pixel
    #     evidence cannot reach.
    #
    #     Where an overlay sits, the picture's high-frequency content is replaced
    #     (opaque) or scaled by (1-a) (semi-transparent). Per pixel that is a weak
    #     signal: a 35%-alpha mark over a noisy frame moves a single pixel by about
    #     1-2 sigma, which no threshold can separate from noise (measured: that
    #     case seeded nothing at all and was missed outright, hit 0/1175). But the
    #     ratio of a small window's energy to a large window's is an AGGREGATE over
    #     hundreds of pixels, so it is decided far below the per-pixel noise. The
    #     picture's own blobs, however strong, keep their ratio at ~1.
    #     The ratio map is itself noisy, so it is smoothed before thresholding and
    #     the detection threshold is looser than the per-component veto below.
    e_large = cv2.blur(energy, (k, k))
    ratio_map = cv2.blur(energy, (_odd(args.collapse_small), _odd(args.collapse_small))) \
        / np.maximum(e_large, 1e-6)
    col_seed = (cv2.blur(ratio_map, (3, 3)) < args.collapse_detect_ratio) & (e_large > flat_knee)
    if not args.collapse_detect:
        col_seed = np.zeros_like(col_seed)
    if args.open_ksize > 0:
        col_seed = cv2.morphologyEx(col_seed.astype(np.uint8) * 255, cv2.MORPH_OPEN,
                                    np.ones((args.open_ksize, args.open_ksize), np.uint8)) > 0
    seed = res_seed | col_seed | seed_mean

    # 3. optional relaxation (--grow-ratio < 1) grows the seed set into weaker
    #    pixels of the SAME relative standing; at the default 1.0 the seed set is
    #    used as is. Growth is deliberately not an absolute threshold: an absolute
    #    one leaks into artwork whose residual is merely high, and once a seed
    #    merges with artwork its fate is decided by the artwork's statistics --
    #    measured: on the busy frame that merged mask covered 97% of the frame and
    #    the mark was dropped along with it.
    grow_thr = np.maximum(args.contrast_delta, args.saliency_ratio * args.grow_ratio * level)
    grow = (best_mag > grow_thr)

    ctx = {
        "allow": allow, "args": args, "best": best, "best_mag": best_mag,
        "energy": energy, "flat_knee": flat_knee, "cap_frac": args.max_area_frac,
        "frame_energy": median_e, "frame_textured": median_e > flat_knee,
    }
    # The two detectors run through SEPARATE component pipelines and their accepted
    # masks are unioned at the end. Running them through one pipeline was measured
    # to destroy the collapse signal: on the low-contrast opaque case the collapse
    # seeds cover 74.5% of the mark and the picture's own residual seeds number
    # ~48000, so a union of the seeds merges the mark into artwork and every
    # component test then judges the mixture. Kept apart, each detector's evidence
    # is judged on its own terms.
    stats_out = []
    mask_res, stats_res = accept_components(
        seed=(res_seed | seed_mean), evidence=(grow | seed_mean), ctx=ctx)
    stats_out.extend(stats_res)
    mask_col = np.zeros_like(mask_res)
    if col_seed.any():
        mask_col, stats_col = accept_components(seed=col_seed, evidence=col_seed, ctx=ctx,
                                                kind="texture-collapse")
        stats_out.extend(stats_col)
    mask = cv2.bitwise_or(mask_res, mask_col)

    mask[allow == 0] = 0
    if shape_mask is not None:
        mask[shape_mask == 0] = 0
    confidence = 0.0
    if mask.any():
        confidence = float(best_mag[mask > 0].mean() / max(floor, 1e-6))
    return mask, {"floor": round(float(floor), 2),
                  "localLevelP50": round(float(np.median(level)), 2),
                  "seedPx": int((res_seed | seed_mean).sum()),
                  "collapseSeedPx": int(col_seed.sum()),
                  "growPx": int(grow.sum()),
                  "components": stats_out,
                  "confidence": round(confidence, 2)}


def accept_components(seed, evidence, ctx, kind="residual"):
    """Turn a seed mask into an accepted mask, one component at a time.

    Judged on the component's EVIDENCE pixels, never on the filled interior: the
    interior of an opaque mark has no residual of its own, so averaging over it
    would dilute a real mark toward zero exactly when the mark is large -- another
    way for a mark to cancel itself out.

    Holes are filled PER COMPONENT and the component is dropped when the fill
    encloses far more than the outline itself. That test exists because a picture
    feature can produce a closed ring of residual too; filling such a ring
    (measured: 20% of the frame) is how an earlier version turned one seed ring
    into a giant painted region. A glyph's outline encloses its own area, at a
    ratio of a few; a ring around artwork encloses orders more.
    """
    args = ctx["args"]
    allow = ctx["allow"]
    best_mag, best, energy = ctx["best_mag"], ctx["best"], ctx["energy"]
    allow_area = max(1, int((allow > 0).sum()))
    cap = ctx["cap_frac"] * allow_area
    stats_out = []
    keep = np.zeros(seed.shape, np.uint8)
    work = seed.astype(np.uint8) * 255
    if not work.any():
        return keep, stats_out
    # bridge small gaps in a mark's outline before judging components: the interior
    # of an opaque mark carries no residual of its own, so it can only be recovered
    # from a CLOSED outline.
    ck = _odd(max(3, args.open_ksize if args.close_ksize <= 0 else args.close_ksize))
    work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, np.ones((ck, ck), np.uint8))

    n, labels, stats, _cent = cv2.connectedComponentsWithStats((work > 0).astype(np.uint8), 8)
    for i in range(1, n):
        seed_area = int(stats[i, cv2.CC_STAT_AREA])
        if seed_area < args.min_seed_blob:
            continue
        x, y, w, h = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                      int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
        comp = labels == i
        filled = _fill_holes(comp[y:y + h, x:x + w])
        fill_ratio = float(filled.sum()) / max(1, seed_area)
        if args.max_fill_ratio > 0 and fill_ratio > args.max_fill_ratio:
            stats_out.append({"seedArea": seed_area, "box": [x, y, w, h], "source": kind,
                              "fillRatio": round(fill_ratio, 1),
                              "dropped": "fill encloses far more than the outline "
                                         "(ring around artwork, not a glyph)"})
            continue
        area = int(filled.sum())
        if area < args.min_blob:
            continue
        if cap > 0 and area > cap:
            stats_out.append({"seedArea": seed_area, "area": area, "box": [x, y, w, h],
                              "source": kind,
                              "dropped": "area > max-area-frac of search area"})
            continue
        ev = comp & evidence
        if not ev.any():
            continue
        r = best[ev]                             # (N,3) residual vectors of the evidence
        mean_vec = r.mean(axis=0)
        strength = float(np.sqrt(float((mean_vec ** 2).sum())))
        if strength < args.core_delta:
            continue
        if args.sign_consistency > 0.0:
            agree = float((r @ mean_vec > 0).mean())
            if agree < args.sign_consistency:
                continue
        else:
            agree = float("nan")
        ring_level, comp_level = _ring_inner(ev, best_mag, [x, y, w, h], args.window, allow)[::-1]
        saliency = comp_level / max(ring_level, 1e-3)
        comp_full = np.zeros(best_mag.shape, bool)
        comp_full[y:y + h, x:x + w] = filled
        e_in, e_ring = _ring_inner(comp_full, energy, [x, y, w, h], args.window, allow,
                                   erode_r=args.collapse_erode)
        # How flat the interior is is judged against the FRAME's own typical texture,
        # not against the immediate ring. Measured reason: on a busy frame the ring
        # around a flat patch of artwork easily out-textures the patch itself (the
        # ring clips a blob's edge, the patch does not), so a ring-relative test
        # accepted ~20000 px of the picture per frame in every case. Judged against
        # the frame, a picture patch keeps its own noise and stays at ~1.0, while an
        # opaque overlay's interior measures 0.0 and a 35%-alpha one ~0.65.
        ref = max(ctx["frame_energy"], 1e-6)
        collapse = e_in / ref
        # A collapse test on a SMALL component is noise-limited: the median energy
        # over a 150-px patch has a spread of roughly 10%, so a fixed ratio threshold
        # admits hundreds of picture patches (measured: 65-77 spurious components per
        # busy frame, ~4600 px on a CLEAN frame). The depth is therefore weighted by
        # the sample size: z = depth * sqrt(N/4), where /4 is the correlation length
        # of the energy field. A picture patch has depth ~0.4 over ~200 px -> z ~3;
        # an opaque mark has depth 1.0 over thousands of px -> z ~20-40.
        depth = max(0.0, 1.0 - float(collapse))
        collapse_z = depth * float(np.sqrt(max(1.0, area / 4.0)))
        flattened = float(collapse) <= args.collapse_ratio
        if kind == "residual":
            # Where the picture carries texture, a region that still carries its own
            # texture is not an overlay -- that is how the picture's own blobs are
            # rejected on a busy frame. Where the picture is flat there is nothing to
            # flatten, and standing out from the ring is the whole evidence.
            if args.min_saliency > 0 and saliency < args.min_saliency:
                stats_out.append({"seedArea": seed_area, "area": area, "box": [x, y, w, h],
                                  "source": kind, "saliency": round(saliency, 2),
                                  "dropped": "saliency < min-saliency (it is the picture, "
                                             "not an overlay)"})
                continue
            if ctx["frame_textured"] and not flattened:
                stats_out.append({"seedArea": seed_area, "area": area, "box": [x, y, w, h],
                                  "source": kind, "collapse": round(float(collapse), 2),
                                  "dropped": "no texture collapse: this region still carries "
                                             "the picture's own texture, so it is not an overlay"})
                continue
        elif not flattened or collapse_z < args.collapse_z:
            stats_out.append({"seedArea": seed_area, "area": area, "box": [x, y, w, h],
                              "source": kind, "collapse": round(float(collapse), 2),
                              "collapseZ": round(collapse_z, 2),
                              "dropped": "collapse not significant: depth x sqrt(size) below "
                                         "--collapse-z"})
            continue
        keep[y:y + h, x:x + w][filled] = True
        stats_out.append({
            "source": kind,
            "seedArea": seed_area,
            "area": area,
            "fillRatio": round(fill_ratio, 2),
            "evidencePx": int(ev.sum()),
            "box": [x, y, w, h],
            "meanResidual": [round(float(v), 2) for v in mean_vec],
            "strength": round(strength, 2),
            "agree": None if agree != agree else round(agree, 3),
            "saliency": round(saliency, 2),
            "evidenceKind": "texture-collapse" if kind == "texture-collapse" else "saliency",
            "collapse": round(float(collapse), 2),
            "collapseZ": round(collapse_z, 2),
            "ringCollapse": round(float(e_in / max(e_ring, 1e-6)), 2),
        })
    return keep, stats_out


def detect_multi(imgs, allow, args):
    """Same mark, many frames: per-pixel spread collapses where the mark sits.

    The raw criterion -- "this pixel varies less across the frames than its
    neighbourhood does" -- also fires on any patch of plain background that all
    the frames happen to share, and unlike the single-image path this one had no
    component validation at all. Measured on 10 real photos sharing one corner
    mark: the mask came out as the mark (11091 px) PLUS a 583 px patch of flat
    background 100 px above it, and every one of those 583 px was painted over in
    all 10 outputs -- 4995 changed pixels of the user's picture, which is exactly
    what this tool promises never to do.

    So each candidate component must show the one thing a watermark has and a patch
    of background does not: **a drawn boundary**. Measured on the same 10 photos,
    the mean gradient magnitude on a component's rim is 3.6 for the background
    patch (below the frame's own median of 8.7) and 63-108 for every glyph -- a
    factor of 20 between them, so this is not a close call.

    It runs on the per-pixel median of the stack, which keeps the mark (present in
    every frame) and cancels the different pictures. The mean-level difference
    between a component and its ring was tried first and is the wrong statistic
    here: this mark's white fill sits on light backgrounds, so its level difference
    is only 9-12 levels while the background patch scores 0 -- any threshold that
    separates those two is luck, whereas the boundary test separates them 20x.
    """
    stack = np.stack([cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32) for im in imgs])
    spread = stack.max(axis=0) - stack.min(axis=0)
    k = args.window * 2 + 1
    local = cv2.blur(spread, (k, k))
    ratio = spread / np.maximum(local, 1e-3)
    raw = ((ratio < args.multi_ratio) & (local > args.multi_min_local)).astype(np.uint8)
    raw[allow == 0] = 0

    median_img = np.median(stack, axis=0)
    grad = cv2.magnitude(cv2.Sobel(median_img, cv2.CV_32F, 1, 0, ksize=3),
                         cv2.Sobel(median_img, cv2.CV_32F, 0, 1, ksize=3))
    frame_grad = float(np.median(grad))
    mask = np.zeros(raw.shape, np.uint8)
    dropped = []
    if raw.any():
        n, labels, stats, _c = cv2.connectedComponentsWithStats(raw, 8)
        k3 = np.ones((3, 3), np.uint8)
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < args.min_blob:
                continue
            x, y, w, h = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                          int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
            comp = labels == i
            sub = comp[y:y + h, x:x + w].astype(np.uint8)
            rim = (cv2.dilate(sub, k3) > 0) & (cv2.erode(sub, k3) == 0)
            edge = float(grad[y:y + h, x:x + w][rim].mean()) if rim.any() else 0.0
            if edge < args.multi_min_edge and edge < args.multi_edge_ratio * frame_grad:
                dropped.append({"area": area, "box": [x, y, w, h], "rimEdge": round(edge, 1),
                                "frameEdge": round(frame_grad, 1),
                                "dropped": "consistent across frames but has no drawn boundary, "
                                           "so it is background the frames share, not an overlay"})
                continue
            mask[comp] = 255
    k2 = np.ones((args.dilate * 2 + 1, args.dilate * 2 + 1), np.uint8)
    mask = cv2.dilate(mask, k2, iterations=1) if mask.any() else mask
    mask[allow == 0] = 0
    return mask, {"multiComponents": dropped}


# --------------------------------------------------------------------------
# Tiled watermarks
# --------------------------------------------------------------------------
def _best_lag(profile, min_lag, max_lag):
    """Smallest strong period of a 1-D projection, by normalized autocorrelation."""
    x = profile.astype(np.float64)
    x = x - x.mean()
    n = x.size
    if n < 8 or not np.any(x):
        return 0, 0.0
    spec = np.fft.rfft(x, n=2 * n)
    ac = np.fft.irfft(spec * np.conj(spec), n=2 * n)[:n]
    if ac[0] <= 0:
        return 0, 0.0
    ac = ac / ac[0]
    max_lag = min(max_lag, n - 2)
    if max_lag <= min_lag:
        return 0, 0.0
    window = ac[min_lag:max_lag]
    lag = int(np.argmax(window)) + min_lag
    score = float(ac[lag])
    # require a local peak, not a shoulder of the zero-lag lobe
    if lag > min_lag and ac[lag] >= ac[lag - 1] and (lag + 1 >= n or ac[lag] >= ac[lag + 1]):
        return lag, score
    return 0, score


def estimate_period(bgr, args, restrict=None):
    """Find a tiled mark's lattice from the residual energy projected on each axis.

    A tiled mark is periodic; the picture under it is not. Projecting the
    residual magnitude onto x and y and autocorrelating each projection finds the
    mark's repeat even when the mark itself is barely visible in any single tile.

    Returns (period, detail). `period` is (0,0) when no confident lattice exists;
    the caller then refuses to guess rather than stamping a wrong grid -- a wrong
    lattice modifies clean pixels, which this tool promises never to do.
    """
    _best, mag, _votes, floor = residual_field(bgr, args)
    ys, xs = np.nonzero(restrict if restrict is not None else np.ones(mag.shape, bool))
    if ys.size:
        sub = mag[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    else:
        sub = mag
    col = sub.sum(axis=0)
    row = sub.sum(axis=1)
    h, w = mag.shape
    px, sx = _best_lag(col, max(8, w // 16), max(16, w // 2))
    py, sy = _best_lag(row, max(8, h // 16), max(16, h // 2))
    return (px, py), {"xScore": round(sx, 3), "yScore": round(sy, 3),
                      "minScore": args.period_min_score}


def folded_lattice_mask(bgr, allow, period, args):
    """Recover the mark's shape inside ONE lattice cell, then replicate it.

    Folding the residual magnitude modulo the period stacks every repetition of
    the mark on top of itself and averages the picture underneath away, so the
    shape is recovered from all tiles at once instead of from the weakest one.
    """
    pw, ph = period
    if pw < 4 or ph < 4:
        return np.zeros(bgr.shape[:2], np.uint8), {"cells": 0}
    _best, mag, _votes, floor = residual_field(bgr, args)
    h, w = mag.shape
    allow_b = allow > 0
    acc = np.zeros((ph, pw), np.float64)
    cnt = np.zeros((ph, pw), np.float64)
    yy, xx = np.nonzero(allow_b)
    np.add.at(acc, (yy % ph, xx % pw), mag[yy, xx])
    np.add.at(cnt, (yy % ph, xx % pw), 1.0)
    cells = float(cnt.mean())
    folded = np.where(cnt > 0, acc / np.maximum(cnt, 1.0), 0.0)
    thr = max(args.contrast_delta, args.k_sigma * float(np.median(folded)))
    shape = (folded > thr).astype(np.uint8) * 255
    if shape.any() and args.open_ksize > 0:
        shape = cv2.morphologyEx(shape, cv2.MORPH_CLOSE,
                                 np.ones((args.open_ksize, args.open_ksize), np.uint8))
        contours, _ = cv2.findContours(shape, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(shape)
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
        shape = filled
        # drop specks that are too small to be a mark stroke
        n, labels, stats, _c = cv2.connectedComponentsWithStats((shape > 0).astype(np.uint8), 8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < args.min_blob // 4:
                shape[labels == i] = 0
    idx_y = np.arange(h) % ph
    idx_x = np.arange(w) % pw
    mask = shape[np.ix_(idx_y, idx_x)].copy()
    mask[allow == 0] = 0
    return mask, {"cells": int(round(cells)), "cellPx": int(shape.sum()),
                  "foldThreshold": round(float(thr), 2), "period": [pw, ph]}


# --------------------------------------------------------------------------
# Strategy A: exact signature (learn / restore)
# --------------------------------------------------------------------------
def _linear_alpha(observed, original, weights=None):
    """Solve `observed = (1-a)*original + a*color` for the scalar a.

    Least squares over the whole sample, closed form. Writing d = observed-original
    (the residual) and u = mean(original) - original, the model says
    d = a*color - a*original, so the part of d that varies with the picture is
    exactly -a*u and does not involve the unknown colour at all:

        a = <d - mean(d), u> / <u, u>

    The colour then follows from the mean:  color = mean(original) + mean(d)/a.
    """
    d = observed.astype(np.float64) - original.astype(np.float64)
    u = original.astype(np.float64).mean(axis=0, keepdims=True) - original.astype(np.float64)
    v = d - d.mean(axis=0, keepdims=True)
    if weights is not None:
        w = weights.astype(np.float64).reshape(-1, 1)
        u, v = u * w, v * w
    denom = float((u * u).sum())
    if denom < 1e-9:
        return 0.0, original.astype(np.float64).mean(axis=0)
    a = float((v * u).sum() / denom)
    mean_d = d.mean(axis=0)
    if a <= 1e-6:
        return 0.0, original.astype(np.float64).mean(axis=0)
    color = original.astype(np.float64).mean(axis=0) + mean_d / a
    return a, color


def learn_signature(pairs, args):
    """Recover (alpha map, colour) from marked/clean pairs of the same picture.

    Two stages, deliberately separated:
      1. a GLOBAL (a, color) per pair from the least-squares fit above, using only
         the pixels the pair disagrees on;
      2. a PER-PIXEL alpha map, by projecting each pixel's residual onto the
         direction (color - original). The global fit gives the colour and the
         overall strength; the per-pixel projection is what makes an antialiased
         or partially transparent mark reproducible instead of a hard shape.
    Pairs are combined by taking the median of the per-pixel maps where they
    overlap, and the reported reconstruction RMSE says whether the model holds at
    all (a non-linear or re-compressed mark will show it immediately).
    """
    if not pairs:
        raise ValueError("--learn needs at least one marked:clean pair")
    alphas = []
    colors = []
    shapes = None
    detail = []
    for wm_path, clean_path in pairs:
        wm = imread_any(wm_path, cv2.IMREAD_UNCHANGED)
        cl = imread_any(clean_path, cv2.IMREAD_UNCHANGED)
        if wm is None or cl is None:
            raise ValueError("cannot read pair %s : %s" % (wm_path, clean_path))
        if wm.shape[:2] != cl.shape[:2]:
            raise ValueError("pair %s : %s have different sizes" % (wm_path, clean_path))
        wm = wm[:, :, :3] if wm.ndim == 3 else cv2.cvtColor(wm, cv2.COLOR_GRAY2BGR)
        cl = cl[:, :, :3] if cl.ndim == 3 else cv2.cvtColor(cl, cv2.COLOR_GRAY2BGR)
        diff = np.abs(wm.astype(np.int16) - cl.astype(np.int16)).max(axis=2)
        touched = diff > args.learn_tol
        if not touched.any():
            detail.append({"pair": [wm_path, clean_path], "touchedPx": 0,
                           "note": "no differing pixels above --learn-tol"})
            continue
        ys, xs = np.nonzero(touched)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        obs = wm[y0:y1, x0:x1][touched[y0:y1, x0:x1]]
        org = cl[y0:y1, x0:x1][touched[y0:y1, x0:x1]]
        a, color = _linear_alpha(obs, org)
        if a <= 1e-3:
            detail.append({"pair": [wm_path, clean_path], "touchedPx": int(touched.sum()),
                           "alpha": 0.0, "note": "pair is not explained by an overlay model"})
            continue
        # per-pixel alpha: residual projected onto (color - original)
        sub_o = wm[y0:y1, x0:x1].astype(np.float64)
        sub_c = cl[y0:y1, x0:x1].astype(np.float64)
        dir_vec = color.reshape(1, 1, 3) - sub_c
        num = ((sub_o - sub_c) * dir_vec).sum(axis=2)
        den = (dir_vec * dir_vec).sum(axis=2)
        amap = np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0)
        amap = np.clip(amap, 0.0, 1.0)
        amap[~touched[y0:y1, x0:x1]] = 0.0
        # model check: put the mark back and compare with the marked image
        recon = (1.0 - amap)[..., None] * sub_c + amap[..., None] * color.reshape(1, 1, 3)
        rms = float(np.sqrt(np.mean((recon - sub_o) ** 2)))
        base = float(np.sqrt(np.mean((sub_o - sub_c) ** 2)))
        colors.append(color)
        alphas.append((y0, x0, amap))
        detail.append({"pair": [wm_path, clean_path], "touchedPx": int(touched.sum()),
                       "alpha": round(a, 4), "colorBGR": [round(float(v), 2) for v in color],
                       "bbox": [x0, y0, x1 - x0, y1 - y0],
                       "reconstructRms": round(rms, 3), "residualRms": round(base, 3),
                       "explained": round(1.0 - rms / base, 4) if base > 0 else None})
    if not alphas:
        raise ValueError("no usable pair (all pairs were unreadable or show no mark)")

    color = np.median(np.stack(colors), axis=0)
    # combine pairs: median where they overlap, max elsewhere
    y0 = min(a[0] for a in alphas)
    x0 = min(a[1] for a in alphas)
    y1 = max(a[0] + a[2].shape[0] for a in alphas)
    x1 = max(a[1] + a[2].shape[1] for a in alphas)
    shapes = np.zeros((y1 - y0, x1 - x0), np.float32)
    for (py, px, amap) in alphas:
        acc = np.zeros_like(shapes)
        acc[py - y0:py - y0 + amap.shape[0], px - x0:px - x0 + amap.shape[1]] = amap
        shapes = np.maximum(shapes, acc)
    return shapes, color, detail, (x0, y0)


def signature_to_png(alpha_map, color):
    """Signature PNG: RGB = the mark's colour, A = per-pixel alpha."""
    h, w = alpha_map.shape[:2]
    img = np.zeros((h, w, 4), np.uint8)
    img[:, :, 0] = int(round(float(color[0])))
    img[:, :, 1] = int(round(float(color[1])))
    img[:, :, 2] = int(round(float(color[2])))
    img[:, :, 3] = np.clip(np.round(alpha_map * 255.0), 0, 255).astype(np.uint8)
    return img


def sidecar_bbox(path):
    """The `bbox` recorded next to a --learn signature (image pixels, x y w h).

    A learned signature is written at the mark's own size, so it carries a
    POSITION that its pixels cannot express. Resizing such a file onto the frame
    would silently stretch the mark across the whole picture -- measured: restore
    scored 41.74 -> 47.13 RMSE against the true clean image, i.e. it made the
    picture worse while reporting success. So the position comes from the sidecar,
    and a size mismatch without one is an error rather than a silent guess.
    """
    meta = os.path.splitext(path)[0] + ".json"
    if not os.path.exists(meta):
        return None
    try:
        with open(meta, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    bbox = data.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        return [int(v) for v in bbox]
    return None


def load_signature(path, shape):
    """Read a signature/template PNG -> (alpha map float32 in [0,1], colour or None).

    A 4-channel PNG carries both (alpha in A, colour in RGB). Anything else is a
    binary shape: alpha then comes from --alpha or is calibrated from the image.
    A signature smaller than the frame is placed at its sidecar bbox; without a
    sidecar that is an error, never a resize (see sidecar_bbox).
    """
    img = imread_any(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] != 4:
        return None, None
    alpha = img[:, :, 3].astype(np.float32) / 255.0
    core = alpha > 0.5 * max(float(alpha.max()), 1e-6)
    color = img[:, :, :3][core].reshape(-1, 3).mean(axis=0) if core.any() else None
    h, w = shape[:2]
    if img.shape[:2] != (h, w):
        bbox = sidecar_bbox(path)
        if bbox is None:
            raise ValueError(
                "signature %s is %dx%d but the image is %dx%d and no .json sidecar gives its "
                "position; pass a full-size template, or a signature written by --learn"
                % (os.path.basename(path), img.shape[1], img.shape[0], w, h))
        x, y, bw, bh = bbox
        canvas = np.zeros((h, w), np.float32)
        sx0, sy0 = max(0, -x), max(0, -y)
        dx0, dy0 = max(0, x), max(0, y)
        cw = min(bw - sx0, w - dx0, img.shape[1] - sx0)
        ch = min(bh - sy0, h - dy0, img.shape[0] - sy0)
        if cw > 0 and ch > 0:
            canvas[dy0:dy0 + ch, dx0:dx0 + cw] = alpha[sy0:sy0 + ch, sx0:sx0 + cw]
        alpha = canvas
    return alpha, (None if color is None else color.astype(np.float32))


def binary_shape(path, shape):
    """A template without alpha -> a binary coverage mask."""
    img = imread_any(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3:
        if img.shape[2] == 4:
            m = img[:, :, 3]
        else:
            m = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        m = img
    if m.shape[:2] != shape[:2]:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.where(m > 127, 255, 0).astype(np.uint8)


def calibrate_from_image(bgr, coverage, args):
    """Estimate (a, color) from ONE image whose mark shape is known.

    The clean twin is unknown, so it is approximated by inpainting the shape
    away, and the same least-squares fit as --learn runs against that estimate.
    This is an estimate, not a measurement: the caller must report the residual
    it produces (the tool prints it) so the user can judge it.
    """
    mask = (coverage > 0.5 * max(coverage.max(), 1e-6)).astype(np.uint8) * 255
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    bg = cv2.inpaint(bgr, mask, args.radius, cv2.INPAINT_TELEA)
    sel = mask > 0
    if int(sel.sum()) < 16:
        return 0.0, None
    a, color = _linear_alpha(bgr[sel], bg[sel])
    return a, color


def restore_exact(bgr, alpha_map, color, args):
    """Invert the overlay exactly: original = (observed - a*color) / (1 - a).

    Pixels the mark covers completely (a ~ 1) have no recoverable original -- the
    division blows up, and no algorithm can undo an opaque cover. Those pixels
    are listed and handed to the inpainter instead, so the output never contains
    a division artifact; the count is reported rather than hidden.
    """
    a = np.clip(alpha_map.astype(np.float32), 0.0, 1.0)
    sel = a > args.alpha_floor
    if not sel.any():
        return bgr.copy(), 0, 0
    color = np.asarray(color, np.float32).reshape(1, 1, 3)
    opaque = a > args.opaque_alpha
    a_safe = np.clip(a, 0.0, args.opaque_alpha)
    obs = bgr.astype(np.float32)
    out = bgr.copy()
    denom = np.maximum(1.0 - a_safe, 1e-3)[..., None]
    solved = (obs - a_safe[..., None] * color) / denom
    out[sel] = np.clip(solved, 0, 255)[sel].astype(np.uint8)
    opaque_n = int(opaque.sum())
    if opaque_n:
        out = cv2.inpaint(out, (opaque.astype(np.uint8) * 255), args.radius, cv2.INPAINT_TELEA)
    return out, int(sel.sum()), opaque_n


# --------------------------------------------------------------------------
# Per-file work
# --------------------------------------------------------------------------
def periodic_mask(shape, period, thickness):
    """Tiled mark fallback: stamp a block on the given lattice."""
    h, w = shape[:2]
    pw, ph = period
    if pw <= 0 or ph <= 0:
        return np.zeros((h, w), np.uint8)
    mask = np.zeros((h, w), np.uint8)
    ty, tx = thickness
    y = 0
    while y < h:
        for x in range(0, w, pw):
            mask[y:min(h, y + ty), x:min(w, x + tx)] = 255
        y += ph
    return mask


def remove_one(path, out_path, args, rects, template=None, stack=None):
    img = imread_any(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return False, "cannot read", {}
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    alpha = None
    if img.shape[2] == 4:
        alpha = img[:, :, 3]
        bgr = img[:, :, :3].copy()
    else:
        bgr = img

    allow = search_mask(bgr.shape, args.search, rects)
    if allow.sum() == 0 and args.search == "none" and not rects and not args.period:
        return False, "nothing to do: --search none needs --rect or --period", {}

    info = {"strategy": args.strategy, "restore": bool(args.restore)}
    sig_alpha = template["alpha"] if template else None
    sig_color = template["color"] if template else None

    if args.period:
        period, per_detail = args.period_pair, {}
        if period == (0, 0):
            raise ValueError("no period")
        mask, per_detail = folded_lattice_mask(bgr, allow, period, args)
        info["period"] = per_detail
        if mask.sum() == 0:  # folding found nothing -> fall back to the block stamp
            mask = periodic_mask(bgr.shape, period, (40, 40))
            mask = cv2.bitwise_and(mask, allow)
            info["periodFallback"] = "block stamp"
    elif stack is not None:
        mask, multi_info = detect_multi(stack, allow, args)
        if multi_info.get('multiComponents'):
            info['multiComponents'] = multi_info['multiComponents']
    elif args.restore and sig_alpha is not None:
        mask = (sig_alpha > args.alpha_floor).astype(np.uint8) * 255
        mask[allow == 0] = 0
    else:
        shape_mask = sig_alpha if sig_alpha is not None else None
        mask, det = detect_single(bgr, allow, args, shape_mask)
        info.update(det)

    if args.dilate > 0 and mask.sum() > 0:
        k = np.ones((args.dilate * 2 + 1, args.dilate * 2 + 1), np.uint8)
        mask = cv2.dilate(mask, k, iterations=1)

    changed = int((mask > 0).sum())
    total = mask.shape[0] * mask.shape[1]
    info["maskPx"] = changed

    out = img
    note = "no watermark pixels matched (nothing written as 'removed')"
    if changed == 0:
        info["changedPct"] = 0.0
        if args.restore and sig_alpha is not None:
            note = "signature matched no pixels in the search area"
    elif args.restore and sig_alpha is not None:
        if sig_color is None:
            a_est, c_est = calibrate_from_image(bgr, sig_alpha, args)
            if c_est is None:
                return False, "cannot calibrate a colour for this signature", info
            color = c_est
            info["calibrated"] = {"alpha": round(float(a_est), 4),
                                  "colorBGR": [round(float(v), 2) for v in c_est]}
            if a_est <= 1e-3:
                return False, "signature carries no alpha and none could be calibrated", info
        else:
            color = sig_color
        fixed, solved_n, opaque_n = restore_exact(bgr, sig_alpha, color, args)
        solvable = (sig_alpha > args.alpha_floor) & (sig_alpha <= args.opaque_alpha)
        fixed_img = bgr.copy()
        if solvable.any():
            fixed_img[solvable] = fixed[solvable]
        if opaque_n:
            fixed_img = cv2.inpaint(fixed_img, ((sig_alpha > args.opaque_alpha).astype(np.uint8) * 255),
                                    args.radius, cv2.INPAINT_TELEA)
        out = np.dstack([fixed_img, alpha]) if alpha is not None else fixed_img
        info["solvedPx"] = int(solvable.sum())
        info["opaquePx"] = opaque_n
        note = ("inverted the signature exactly on %d px (%.3f%% of frame); "
                "%d px were opaque and inpainted instead" % (
                    int(solvable.sum()), 100.0 * changed / total, opaque_n))
    else:
        flag = cv2.INPAINT_TELEA if args.mode == "telea" else cv2.INPAINT_NS
        fixed = cv2.inpaint(bgr, mask, args.radius, flag)
        out = np.dstack([fixed, alpha]) if alpha is not None else fixed
        note = "inpainted %d px (%.3f%% of frame)" % (changed, 100.0 * changed / total)

    info["changedPct"] = round(100.0 * changed / total, 4)

    if args.mask_out_dir:
        os.makedirs(args.mask_out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(path))[0]
        imwrite_any(os.path.join(args.mask_out_dir, stem + "-mask.png"), mask)
    if args.dry_run:
        return True, note + " [dry run]", info
    if os.path.exists(out_path) and not args.overwrite:
        return False, "exists (use --overwrite): %s" % out_path, info
    ok = imwrite_any(out_path, out)
    return ok, note, info


def collect_inputs(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            for root, _d, names in os.walk(p):
                if os.path.basename(root) in ("clean", "masks"):
                    continue
                for n in sorted(names):
                    if n.lower().endswith(IMAGE_EXT) and "-clean" not in n:
                        files.append(os.path.join(root, n))
        elif os.path.isfile(p):
            files.append(p)
    return files


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(description="General batch watermark removal.")
    ap.add_argument("inputs", nargs="*", help="files or directories (walked)")
    ap.add_argument("-o", "--outdir", default="")
    ap.add_argument("--search", default="all", choices=list(SEARCH.keys()),
                    help="where to look (default all = whole frame)")
    ap.add_argument("--rect", action="append", default=[], help="x,y,w,h (repeatable)")
    ap.add_argument("--period", default="",
                    help="W,H lattice of a tiled mark, or 'auto' to estimate it")
    ap.add_argument("--period-min-score", type=float, default=0.25,
                    help="autocorrelation peak required before 'auto' accepts a lattice")
    ap.add_argument("--template", default="", help="mark PNG (alpha) or binary shape")
    ap.add_argument("--restore", action="store_true",
                    help="invert the overlay exactly instead of inpainting")
    ap.add_argument("--alpha", type=float, default=-1.0,
                    help="peak alpha of a binary template (default: calibrate from the image)")
    ap.add_argument("--alpha-floor", type=float, default=1.0 / 255.0,
                    help="ignore signature pixels weaker than this")
    ap.add_argument("--opaque-alpha", type=float, default=0.97,
                    help="above this the cover is treated as opaque and inpainted, not solved")
    ap.add_argument("--color", default="", help="B,G,R of the mark (default: from signature/calibration)")
    ap.add_argument("--learn", action="append", default=[], nargs=2, metavar=("MARKED", "CLEAN"),
                    help="a marked/clean pair of the SAME picture, used to solve the mark's "
                         "alpha and colour (repeatable). Two arguments, not one 'a:b' string: "
                         "on Windows a path already contains a colon")
    ap.add_argument("--signature-out", default="",
                    help="where --learn writes the signature PNG (also .json sidecar)")
    ap.add_argument("--learn-tol", type=int, default=1,
                    help="min per-channel difference for a pixel to count as covered")
    ap.add_argument("--strategy", default="auto", choices=["auto", "single", "multi", "template"])
    # detector knobs
    ap.add_argument("--contrast-delta", type=float, default=12.0,
                    help="absolute floor on the residual magnitude")
    ap.add_argument("--k-sigma", type=float, default=3.0,
                    help="also require this many robust noise sigmas (frame-wide estimate). "
                         "Raise it on a noisy frame, lower it for a very faint mark")
    ap.add_argument("--auto-scales", dest="auto_scales", action="store_true", default=False,
                    help="also use an image-relative background scale (off by default: "
                         "frame-scale windows turn a picture's own soft blobs into residuals)")
    ap.add_argument("--no-auto-scales", dest="auto_scales", action="store_false",
                    help="only use --window-derived scales (default)")
    ap.add_argument("--max-area-frac", type=float, default=0.25,
                    help="drop a component larger than this share of the search area "
                         "(a watermark is an overlay, not a quarter of the picture); 0 disables")
    ap.add_argument("--max-fill-ratio", type=float, default=6.0,
                    help="drop a component whose enclosed area exceeds this multiple of its own "
                         "outline (a ring traced around artwork, not a glyph); 0 disables")
    ap.add_argument("--min-seed-blob", type=int, default=12,
                    help="drop a seed component smaller than this before filling")
    ap.add_argument("--collapse-ratio", type=float, default=0.75,
                    help="where the surroundings carry texture, the component's interior must "
                         "hold at most this share of it (an overlay flattens texture)")
    ap.add_argument("--collapse-detect-ratio", type=float, default=0.8,
                    help="looser threshold used when collapse acts as a DETECTOR rather than a "
                         "veto (the ratio map is smoothed first, so this needs headroom)")
    ap.add_argument("--collapse-z", type=float, default=6.0,
                    help="significance required of a collapse-detected component: "
                         "(1 - collapse) * sqrt(area/4) must reach this; 0 disables. "
                         "Measured on the suite: real marks score 21-38, the picture patches "
                         "that survive the detector score 4.1-4.9, and a clean busy frame "
                         "yields 0 px at 4.0 already")
    ap.add_argument("--collapse-erode", type=int, default=2,
                    help="pixels to erode before measuring the interior texture (keeps the "
                         "mark's own antialiased edge out of the measurement)")
    ap.add_argument("--flat-energy", type=float, default=4.0,
                    help="absolute texture energy below which a neighbourhood counts as flat")
    ap.add_argument("--flat-energy-frac", type=float, default=0.5,
                    help="same, as a share of the frame's median texture energy")
    ap.add_argument("--collapse-small", type=int, default=5,
                    help="window used for the 'inside' texture energy of the collapse detector "
                         "(must stay below the mark's stroke width)")
    ap.add_argument("--collapse-detect", dest="collapse_detect", action="store_true",
                    default=True,
                    help="use texture collapse as a detector as well as an acceptance test "
                         "(its own component pipeline; disable for a picture with large "
                         "genuinely flat areas, where a flat region is not an overlay)")
    ap.add_argument("--no-collapse-detect", dest="collapse_detect", action="store_false",
                    help=argparse.SUPPRESS)
    ap.add_argument("--match-window", type=int, default=0,
                    help="window of the matched-filter seed (mean residual vector); keep it near "
                         "the mark's stroke width; 0 disables (default). Measured: it preserves "
                         "the picture's OWN coherent structure as well as the mark's, so it needs "
                         "the same local normalisation to be useful and was net-negative on the "
                         "suite as shipped")
    ap.add_argument("--match-gain", type=float, default=1.0,
                    help="how much more than the noise-reduced floor a matched-filter seed needs")
    ap.add_argument("--grow-ratio", type=float, default=1.0,
                    help="relative threshold used to grow the seed set (1.0 = seeds only; "
                         "lower it for a faint mark whose outline is broken)")
    ap.add_argument("--close-ksize", type=int, default=0,
                    help="kernel used to bridge gaps in a mark's outline (0 = follow --open-ksize)")
    ap.add_argument("--min-votes", type=int, default=1,
                    help="how many background scales must clear the floor at a pixel")
    ap.add_argument("--sign", default="both", choices=["both", "bright", "dark"],
                    help="kept for compatibility; the vector test is sign-agnostic")
    ap.add_argument("--window", type=int, default=31, help="base local window radius")
    ap.add_argument("--min-blob", type=int, default=40, help="drop components smaller than this (px)")
    ap.add_argument("--core-delta", type=float, default=6.0,
                    help="min |mean residual vector| of a component")
    ap.add_argument("--sign-consistency", type=float, default=0.7,
                    help="min share of a component's pixels agreeing with its mean direction")
    ap.add_argument("--min-saliency", type=float, default=5.0,
                    help="component residual level must exceed its own ring by this "
                         "factor (rejects artwork that merged with something sharp); 0 disables. "
                         "Measured separation on the suite: real marks score 9.8-10.3, the "
                         "artwork components that survive every other filter score 2.3-4.0")
    ap.add_argument("--saliency-ratio", type=float, default=1.8,
                    help="per-pixel seed/grow threshold: how far above the local residual "
                         "level a pixel must be; 0 falls back to the absolute floor only")
    ap.add_argument("--open-ksize", type=int, default=3)
    # legacy knobs accepted so old command lines keep working
    ap.add_argument("--sat-delta", type=float, default=30.0, help=argparse.SUPPRESS)
    ap.add_argument("--sat-min", type=float, default=60.0, help=argparse.SUPPRESS)
    ap.add_argument("--bright-delta", type=float, default=25.0, help=argparse.SUPPRESS)
    ap.add_argument("--edge-delta", type=float, default=0.0, help=argparse.SUPPRESS)
    ap.add_argument("--no-require-core", dest="require_core", action="store_false",
                    help=argparse.SUPPRESS)
    # multi-image knobs
    ap.add_argument("--multi-ratio", type=float, default=0.55)
    ap.add_argument("--multi-min-local", type=float, default=6.0)
    ap.add_argument("--multi-min-edge", type=float, default=12.0,
                    help="min mean gradient magnitude on a multi-image candidate's rim: a "
                         "watermark has a drawn boundary, a patch of background the frames "
                         "happen to share does not (measured 63-108 vs 3.6 on real photos)")
    ap.add_argument("--multi-edge-ratio", type=float, default=3.0,
                    help="same test relative to the frame's own median gradient, for pictures "
                         "whose texture is much stronger or weaker than the sample")
    # output / inpainting
    ap.add_argument("--dilate", type=int, default=3)
    ap.add_argument("--radius", type=int, default=5)
    ap.add_argument("--mode", default="telea", choices=["telea", "ns"])
    ap.add_argument("--suffix", default="-clean")
    ap.add_argument("--out-ext", default="",
                    help="output extension (default: the input's). Use png when the input is a "
                         "JPEG: writing JPEG would re-compress every pixel of the picture, so "
                         "the output would differ everywhere and no longer be a record of what "
                         "the tool actually changed")
    ap.add_argument("--mask-out-dir", default="")
    ap.add_argument("--json-out", default="", help="write a machine-readable run report here")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap


def run_learn(args):
    pairs = []
    for spec in args.learn:
        if len(spec) != 2:
            print("[FAIL] --learn needs two paths (marked, clean); got %r" % (spec,))
            return 2
        pairs.append((spec[0], spec[1]))
    try:
        alpha_map, color, detail, origin = learn_signature(pairs, args)
    except ValueError as exc:
        print("[FAIL] %s" % exc)
        return 2
    out = args.signature_out or "watermark-signature.png"
    png = signature_to_png(alpha_map, color)
    if not imwrite_any(out, png):
        print("[FAIL] cannot write %s" % out)
        return 2
    ys, xs = np.nonzero(alpha_map > args.alpha_floor)
    bbox = [int(xs.min()) + origin[0], int(ys.min()) + origin[1],
            int(xs.max() - xs.min()) + 1, int(ys.max() - ys.min()) + 1] if xs.size else None
    meta = {
        "schema": SIGNATURE_SCHEMA,
        "kind": "alpha-map",
        "alphaFile": os.path.basename(out),
        "colorBGR": [round(float(v), 3) for v in color],
        "colorRGBHex": "#%02x%02x%02x" % (
            int(round(min(255, max(0, color[2])))), int(round(min(255, max(0, color[1])))),
            int(round(min(255, max(0, color[0]))))),
        "alphaPeak": round(float(alpha_map.max()), 4),
        "alphaMeanOnMark": round(float(alpha_map[alpha_map > args.alpha_floor].mean()), 4)
        if (alpha_map > args.alpha_floor).any() else 0.0,
        "alphaPixels": int((alpha_map > args.alpha_floor).sum()),
        "bbox": bbox,
        "pairs": detail,
        "usage": "--template %s --restore" % os.path.basename(out),
    }
    side = os.path.splitext(out)[0] + ".json"
    with open(side, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    print(">>> learned signature from %d pair(s)" % len(detail))
    for d in detail:
        print("    %s" % json.dumps(d, ensure_ascii=False))
    print("    color BGR %s  alpha peak %.4f  mark px %d" % (
        [round(float(v), 1) for v in color], alpha_map.max(), meta["alphaPixels"]))
    print("    wrote %s + %s" % (out, side))
    print("[SUMMARY] ok 1 / fail 0 / total 1")
    return 0


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.learn:
        return run_learn(args)

    if not args.inputs:
        print("[FAIL] no inputs given")
        return 2

    rects = []
    try:
        rects = parse_pairs(args.rect)
        if args.color:
            args.color_bgr = parse_color(args.color)
        else:
            args.color_bgr = None
        args.period_pair = (0, 0)
        if args.period and args.period != "auto":
            args.period_pair = parse_period(args.period)
    except ValueError as exc:
        print("[FAIL] %s" % exc)
        return 2

    files = collect_inputs(args.inputs)
    if not files:
        print("[FAIL] no images found in: %s" % ", ".join(args.inputs))
        return 2

    strategy = args.strategy
    if strategy == "auto":
        strategy = "template" if args.template else "single"
    args.strategy = strategy
    if args.period == "auto" and strategy == "multi":
        print("[FAIL] --period auto does not combine with --strategy multi")
        return 2

    template = None
    if args.template:
        probe = imread_any(files[0], cv2.IMREAD_UNCHANGED)
        if probe is None:
            print("[FAIL] cannot read %s" % files[0])
            return 2
        try:
            sig_alpha, sig_color = load_signature(args.template, probe.shape)
        except ValueError as exc:
            print("[FAIL] %s" % exc)
            return 2
        if sig_alpha is not None:
            template = {"alpha": sig_alpha, "color": sig_color, "kind": "alpha"}
        else:
            shape = binary_shape(args.template, probe.shape)
            if shape is None:
                print("[FAIL] cannot read --template %s" % args.template)
                return 2
            cov = shape.astype(np.float32) / 255.0
            if args.alpha >= 0:
                cov = cov * float(args.alpha)
            template = {"alpha": cov, "color": args.color_bgr, "kind": "shape"}
        if strategy == "auto":
            strategy = "template"
    if strategy == "template" and template is None:
        print("[FAIL] --strategy template needs --template <png>")
        return 2
    if args.restore and template is None:
        print("[FAIL] --restore needs --template (the mark's shape/alpha)")
        return 2

    stack = None
    if strategy == "multi":
        stack_imgs = []
        for f in files:
            im = imread_any(f, cv2.IMREAD_UNCHANGED)
            if im is None:
                continue
            if im.ndim == 2:
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            if im.shape[2] == 4:
                im = im[:, :, :3]
            stack_imgs.append(im)
        if len(stack_imgs) < 3:
            print("[FAIL] --strategy multi needs >=3 readable images (got %d)" % len(stack_imgs))
            return 2
        hw = {im.shape[:2] for im in stack_imgs}
        if len(hw) != 1:
            print("[FAIL] --strategy multi needs all images the same size")
            return 2
        stack = stack_imgs
        print("    (multi: stacking %d frames of size %s)" % (len(stack), stack[0].shape[:2]))

    if args.period == "auto":
        probe = imread_any(files[0], cv2.IMREAD_UNCHANGED)
        if probe is None:
            print("[FAIL] cannot read %s" % files[0])
            return 2
        pb = probe[:, :, :3] if probe.ndim == 3 else cv2.cvtColor(probe, cv2.COLOR_GRAY2BGR)
        allow = search_mask(pb.shape, args.search, rects)
        (px, py), detail = estimate_period(pb, args, allow)
        ok = px > 0 and py > 0 and detail["xScore"] >= args.period_min_score \
            and detail["yScore"] >= args.period_min_score
        print("    (auto period: x=%d score=%.3f | y=%d score=%.3f -> %s)" % (
            px, detail["xScore"], py, detail["yScore"],
            "accepted" if ok else "REJECTED (pass --period W,H)"))
        if not ok:
            print("[FAIL] no confident lattice found; refusing to guess "
                  "(a wrong lattice would modify clean pixels)")
            return 1
        args.period_pair = (px, py)
        args.period = "%d,%d" % (px, py)

    print(">>> remove_watermark: %d file(s) | strategy=%s search=%s rects=%s%s%s%s" % (
        len(files), strategy, args.search, rects or "-",
        " period=%s" % args.period if args.period else "",
        " [RESTORE]" if args.restore else "",
        " [DRY RUN]" if args.dry_run else ""))

    ok_n = fail_n = 0
    report = {"schema": REPORT_SCHEMA, "strategy": strategy, "search": args.search,
              "restore": bool(args.restore), "dryRun": bool(args.dry_run),
              "period": args.period or None, "files": []}
    for src in files:
        base, ext = os.path.splitext(os.path.basename(src))
        if args.out_ext:
            ext = args.out_ext if args.out_ext.startswith(".") else "." + args.out_ext
        out_dir = args.outdir or os.path.join(os.path.dirname(src), "clean")
        out_path = os.path.join(out_dir, base + args.suffix + ext)
        if not args.dry_run:
            os.makedirs(out_dir, exist_ok=True)
        try:
            ok, note, info = remove_one(src, out_path, args, rects, template, stack)
        except ValueError:
            print("[FAIL] %-28s no usable period" % os.path.basename(src))
            ok, note, info = False, "no usable period", {}
        entry = {"input": src, "output": None if args.dry_run else out_path,
                 "ok": bool(ok), "note": note}
        entry.update({k: v for k, v in info.items() if k != "strategy"})
        report["files"].append(entry)
        if ok:
            ok_n += 1
            print("    [OK]   %-28s %s" % (os.path.basename(src), note))
        else:
            fail_n += 1
            print("    [FAIL] %-28s %s" % (os.path.basename(src), note))

    report["summary"] = {"ok": ok_n, "fail": fail_n, "total": len(files)}
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
    print("")
    print("[SUMMARY] ok %d / fail %d / total %d" % (ok_n, fail_n, len(files)))
    return 0 if fail_n == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
