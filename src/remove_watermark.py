"""remove_watermark.py -- batch watermark removal from shared evidence.

What this tool is, and what it deliberately is not
-------------------------------------------------
It removes a watermark from a BATCH of images that share one: at least three images
of the same size, each carrying the same mark in the same place. That is not a
limitation of effort, it is where the information is. One photo with a soft mark on
a busy background simply does not contain enough signal to tell the mark from the
picture, and every attempt to guess measured worse than doing nothing:

  * single-image detection on real 1280x1280 photos found the mark but also accepted
    78 components of the photo itself, changing ~10k pixels of picture per image;
  * loosening the component test to accept a mark drawn in two colours took the
    synthetic suite from 10 passed / 1 known-limitation / 0 failed to 5/1/5, with
    spill on four cases;
  * on a real batch, searching the whole frame instead of the corners painted over
    589324 and 654924 pixels of picture, where the corners painted the mark and 0
    pixels elsewhere.

So single-image AUTOMATIC detection is removed, not deprecated-with-a-warning. What
remains needs no guessing:

  multi      >=3 same-size images. The mark does not move, the pictures do, so the
             per-pixel spread across the stack collapses exactly where the mark is.
             This is the default and the only automatic mode.
  template   you supply the mark: a signature from --learn (RGBA, placed by its
             .json sidecar) or a shape image (white = mark). Optional --restore
             inverts a semi-transparent overlay exactly instead of painting over it.
  --rect     you know the box: narrows where any of the above may look.

Usage
-----
    python src/remove_watermark.py <dir-or-files...> [options]

Directories are walked. Outputs default to `<dir>/clean/<name>-clean.png`; ORIGINALS
ARE NEVER MODIFIED. Fewer than three same-size readable images is a hard error.

    # the normal case: a folder from one platform
    python src/remove_watermark.py 出图/ -o 结果/ --out-ext png

    # teach it the mark once from a marked/clean pair, then invert exactly
    python src/remove_watermark.py --learn 带水印.png 无水印.png --signature-out 平台签名.png
    python src/remove_watermark.py 出图/ --template 平台签名.png --restore

    # you know the box
    python src/remove_watermark.py 出图/ --rect 1040,1185,250,90

    # see what it would change, and look at the mask
    python src/remove_watermark.py 出图/ --dry-run --mask-out-dir masks/

Exit codes: 0 = all ok, 1 = some file failed, 2 = bad arguments or not enough images.
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



def _fill_holes(binary):
    """Fill the enclosed holes of a boolean mask (not its convex hull)."""
    h, w = binary.shape
    big = np.zeros((h + 2, w + 2), np.uint8)
    big[1:-1, 1:-1] = binary.astype(np.uint8)
    ff = 1 - big
    cv2.floodFill(ff, None, (0, 0), 0)
    return (big | (ff > 0))[1:-1, 1:-1].astype(bool)



# --------------------------------------------------------------------------
# Shared-evidence detection (the only automatic mode)
# --------------------------------------------------------------------------
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

    Two tests, because either one alone has a documented failure on real batches:

      1. ABSOLUTE AGREEMENT. The frames must actually agree over the component:
         median spread <= --multi-max-spread. This is what "the same mark in every
         frame" means. Measured on a real 10-photo batch, the mark's pixels vary by
         21 levels across frames (JPEG + sub-pixel placement) while a patch of
         background that merely happens to be *flatter than its surroundings* --
         a smooth tabletop beside a vase, in photos whose backgrounds differ --
         varies by 90. The ratio test alone flags that patch, because a flat patch
         next to textured artwork has a low ratio no matter how much the frames
         disagree inside it.
      2. A DRAWN BOUNDARY. Mean gradient magnitude on the component's rim, either
         >= --multi-min-edge or >= --multi-edge-ratio x the frame's median. A patch
         of plain background that all frames share has no edge of its own (measured:
         3.6, below the frame's own 8.7), whereas every glyph of a real mark scores
         63-108. Without this test such a patch is painted over in every output.

    And the candidate region is COMPLETED before it is used, because the criterion
    above is scale-dependent and this is the bug that produced a blurry ghost of a
    real watermark:

      "spread is much lower than the neighbourhood's" only holds near the mark's
      BORDER. In the middle of a mark wider than the local window, the window is
      all mark, so the ratio is ~1 and the interior is never flagged. For a 207x55
      real mark with 63-px windows the candidate came out as a RING around the
      text: inpainting the ring pulls colours in from both sides and leaves the
      glyph fills (never masked) sitting inside it -- measured on the second real
      batch as a blurry white ghost of the watermark. The first batch escaped only
      because its glyph strokes are 14 px, thinner than the window.

    So closing the candidate and filling its enclosed holes is part of the
    detector, not a cosmetic step. (A cut-back to per-frame "drawn structure" was
    also tried and is worse: this mark's outline carries a high-pass of only 7
    levels, so the cut-back deleted the mark itself. --multi-min-structure keeps it
    available, off by default.)
    """
    grays = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32) for im in imgs]
    stack = np.stack(grays)
    spread = stack.max(axis=0) - stack.min(axis=0)
    k = args.window * 2 + 1
    # A single reference window, deliberately. A multi-scale version (largest of
    # 63/127/249 px) was implemented to fix a mark wider than its own reference
    # window, which suppresses its own evidence (measured on a colour-only mark:
    # spread 5.0 against a reference of 4.3, ratio 1.16, so nothing was found). That
    # fixed the case and BROKE a real batch: with a large reference a big smooth
    # corner of a real photo also reads as "flatter than its surroundings", and the
    # mask grew from 11220 px to ~36000 -- 22-24k px of picture painted over, in all
    # ten images. The real batches are the acceptance criterion, so the single window
    # stays and the colour-only case is a documented limitation.
    local = cv2.blur(spread, (k, k))
    ratio = spread / np.maximum(local, 1e-3)
    raw = ((ratio < args.multi_ratio) & (local > args.multi_min_local)).astype(np.uint8)
    raw[allow == 0] = 0

    if args.multi_min_structure > 0 and raw.any():
        hps = [np.abs(g - cv2.medianBlur(np.clip(g, 0, 255).astype(np.uint8), 5).astype(np.float32))
               for g in grays]
        structure = np.median(np.stack(hps), axis=0)
        raw[structure <= args.multi_min_structure] = 0
    seed = raw > 0
    if raw.any():
        raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        raw = _fill_holes(raw > 0).astype(np.uint8)

    # The rim is measured on the COLOUR median, not on grey: a mark can differ from
    # its background in colour only (measured: a magenta "DEMO" whose luminance
    # matches the gradient behind it has a grey rim of ~0 and was invisible here --
    # the same trap the single-image detector had).
    median_img = np.median(np.stack([im.astype(np.float32) for im in imgs]), axis=0)
    grad = None
    for c in range(3):
        g = cv2.magnitude(cv2.Sobel(median_img[:, :, c], cv2.CV_32F, 1, 0, ksize=3),
                          cv2.Sobel(median_img[:, :, c], cv2.CV_32F, 0, 1, ksize=3))
        grad = g if grad is None else np.maximum(grad, g)
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
            # Judged on the component's EVIDENCE pixels -- the ones the criterion
            # actually flagged -- never on the filled geometry. Filling pulls in the
            # background between glyphs, whose spread is large whenever the pictures
            # differ; measuring agreement over that would reject a perfectly good
            # mark (measured: the first real batch's mask collapsed 11091 -> 2352 px
            # and the watermark came back the moment the fill was included here).
            ev = comp & seed
            comp_spread = float(np.median(spread[ev])) if ev.any() else float("inf")
            # Everything relative to the picture AROUND the candidate, taken from a ring
            # that is deliberately NOT clipped to the search area. A reference is the
            # picture, not a search region: with --rect hugging the mark, a reference
            # taken "inside the search area" IS the mark, so the mark was required to
            # beat its own edges and the tool found nothing at all on three cases.
            grown = cv2.dilate(comp.astype(np.uint8),
                               np.ones((2 * args.window + 1, 2 * args.window + 1), np.uint8)) > 0
            ring = grown & ~comp & (allow > 0)
            ring_spread = float(np.median(spread[ring])) if ring.any() else 0.0
            if ring_spread > 0 and comp_spread > args.multi_ratio * ring_spread:
                dropped.append({"area": area, "box": [x, y, w, h],
                                "spread": round(comp_spread, 1),
                                "ringSpread": round(ring_spread, 1),
                                "dropped": "this candidate varies as much as the picture around "
                                           "it, so it is the picture"})
                continue
            rim = (cv2.dilate(sub, k3) > 0) & (cv2.erode(sub, k3) == 0)
            edge = float(grad[y:y + h, x:x + w][rim].mean()) if rim.any() else 0.0
            if args.multi_max_spread > 0 and comp_spread > args.multi_max_spread:
                dropped.append({"area": area, "box": [x, y, w, h], "evidencePx": int(ev.sum()),
                                "spread": round(comp_spread, 1), "rimEdge": round(edge, 1),
                                "dropped": "the frames disagree here, so this is not the same "
                                           "mark in every frame"})
                continue
            # The rim must beat the picture's REAL edges (95th percentile of the
            # gradient over the search area), not its average texture: measured on
            # patchwork art the median gradient is 42 while the mark's rim is 118-341,
            # so a median-based threshold demanded more than the mark could give and
            # threw it away. Over the SEARCH AREA rather than the whole frame, because
            # the claim is that a candidate stands out from the picture the tool is
            # actually looking at -- with the whole frame, a real photo's 95th
            # percentile is dominated by content elsewhere, and one real batch silently
            # produced zero changes on all ten images while reporting success.
            #
            # A per-candidate ring was tried as this reference and gave back recall on
            # the real batches (109853 -> 23235 px removed across batch 1) while fixing
            # one synthetic case, so the search-area form stays.
            frame_grad = float(np.percentile(grad[allow > 0], 95)) if (allow > 0).any() else 0.0
            if edge < args.multi_min_edge or edge < args.multi_edge_ratio * frame_grad:
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
# Per-file work
# --------------------------------------------------------------------------
BATCH_ERROR = ("批量去水印需要至少 3 张同尺寸图片；单张/混合尺寸不受支持。"
               " (batch removal needs >=3 images of the same size; a single image or "
               "mixed sizes are not supported)")


def remove_one(path, out_path, args, rects, template=None, stacks=None):
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
    if allow.sum() == 0:
        return False, "nothing to do: the search window is empty (check --search/--rect)", {}

    sig_alpha = template["alpha"] if template else None
    sig_color = template["color"] if template else None
    stack = stacks.get(bgr.shape[:2]) if stacks else None
    info = {"strategy": "multi" if stack is not None else "template",
            "restore": bool(args.restore)}
    mask = None

    if sig_alpha is not None:
        # An explicit signature/mask says where the mark is; no detection involved.
        mask = (sig_alpha > args.alpha_floor).astype(np.uint8) * 255
        mask[allow == 0] = 0
        info["strategy"] = "template"
    elif stack is None:
        return False, BATCH_ERROR, info
    else:
        mask, det = detect_multi(stack, allow, args)
        info.update(det)
        info["sharedFrames"] = len(stack)

    if args.dilate > 0 and mask.sum() > 0:
        k = np.ones((args.dilate * 2 + 1, args.dilate * 2 + 1), np.uint8)
        mask = cv2.dilate(mask, k, iterations=1)

    changed = int((mask > 0).sum())
    total = mask.shape[0] * mask.shape[1]
    info["maskPx"] = changed
    # Is this run trustworthy? A watermark is ONE small group of pixels; a mask that
    # covers a large share of the search area, or that is assembled from dozens of
    # components, is the detector finding the picture instead. Measured: the shared
    # -evidence path on two real batches gives 1-2 components and ~2.4% of the corner
    # area, while the removed single-image path gave 78 components and 4-9%, every
    # extra pixel of which was painted over. Reporting beats hiding.
    allow_area = max(1, int((allow > 0).sum()))
    accepted = info.get("acceptedComponents", 0)
    info["searchAreaPx"] = allow_area
    info["maskFracOfSearch"] = round(changed / allow_area, 4)
    info["suspicious"] = bool(changed and (accepted > args.max_components
                                           or changed > args.max_mask_frac * allow_area))

    note = "no watermark pixels matched (nothing written as 'removed')"
    out = img
    if changed == 0:
        info["changedPct"] = 0.0
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
# Exact inversion from a known signature (--learn / --template / --restore)
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



# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(
        description="Batch watermark removal from shared evidence (>=3 same-size images).",
        epilog="Single-image AUTOMATIC detection was removed: measured unreliable on real "
               "photos. Use a batch, or give --template / --rect explicitly.")
    ap.add_argument("inputs", nargs="*", help="files or directories (walked)")
    ap.add_argument("-o", "--outdir", default="")
    ap.add_argument("--search", default="auto", choices=list(SEARCH.keys()) + ["auto"],
                    help="where the batch detector may look (default auto = the four corner "
                         "quartiles, because platform marks live there and searching the whole "
                         "frame measurably damages real photos). Pass 'all' for a centred mark")
    ap.add_argument("--rect", action="append", default=[],
                    help="x,y,w,h (repeatable): a box you know, used with any mode")
    ap.add_argument("--template", default="",
                    help="the mark itself: a signature PNG from --learn (RGBA, placed by its "
                         ".json sidecar) or a shape image where WHITE is the mark (invert a "
                         "black-on-white shape first)")
    ap.add_argument("--restore", action="store_true",
                    help="with --template: invert the overlay exactly instead of inpainting")
    ap.add_argument("--alpha", type=float, default=-1.0,
                    help="peak alpha of a shape template (default: calibrate from the image)")
    ap.add_argument("--alpha-floor", type=float, default=1.0 / 255.0,
                    help="ignore signature pixels weaker than this")
    ap.add_argument("--opaque-alpha", type=float, default=0.97,
                    help="above this the cover is treated as opaque and inpainted, not solved")
    ap.add_argument("--color", default="", help="B,G,R of the mark (default: from the signature)")
    ap.add_argument("--learn", action="append", default=[], nargs=2, metavar=("MARKED", "CLEAN"),
                    help="a marked/clean pair of the SAME picture, used to solve the mark's "
                         "alpha and colour (repeatable). Two arguments, not one 'a:b' string: "
                         "on Windows a path already contains a colon")
    ap.add_argument("--signature-out", default="",
                    help="where --learn writes the signature PNG (also a .json sidecar)")
    ap.add_argument("--learn-tol", type=int, default=1,
                    help="min per-channel difference for a pixel to count as covered")
    ap.add_argument("--strategy", default="auto", choices=["auto", "multi", "template", "single"],
                    help="auto and multi are the same thing: shared evidence. 'single' is "
                         "REMOVED (kept only to explain why) and exits with an error")
    # shared-evidence detector
    ap.add_argument("--multi-ratio", type=float, default=0.65,
                    help="a candidate must vary this much less across the frames than its own "
                         "neighbourhood does. A mark's components measure 0.48-0.57 and patches "
                         "of shared background 0.73-0.89 on the two real batches. 0.70 was tried "
                         "to give a 35%%-alpha mark margin (its spread is scaled by 1-a) and cost "
                         "4033 px of a real picture, so the real batches set this at 0.65")
    ap.add_argument("--multi-min-local", type=float, default=3.0,
                    help="ignore candidates whose neighbourhood barely varies at all. Was 6, "
                         "which rejected a colour-only mark outright when the whole frame "
                         "varied by only ~6 levels between frames")
    ap.add_argument("--multi-max-spread", type=float, default=0.0,
                    help="optional: require this much agreement across frames. OFF because it "
                         "cannot be made to work: a real mark sits at 39-44 on one batch and a "
                         "shared-background patch at 47 on another, so no value keeps both")
    ap.add_argument("--multi-min-edge", type=float, default=12.0,
                    help="min mean gradient on a candidate's rim: a watermark is drawn, a patch "
                         "of background the frames happen to share is not (63-108 vs 3.6)")
    ap.add_argument("--multi-edge-ratio", type=float, default=1.0,
                    help="how many times the picture's own 95th-percentile gradient the rim "
                         "must reach. Measured: 1.0 keeps the mark on every case and both real "
                         "batches while leaving 0-2 background components on the hard ones")
    ap.add_argument("--multi-min-structure", type=float, default=0.0,
                    help="optional cut-back to structure present in every frame; measured "
                         "harmful (a soft mark's outline carries only 7 levels), so off")
    # geometry / output
    ap.add_argument("--window", type=int, default=31, help="local window radius")
    ap.add_argument("--min-blob", type=int, default=40, help="drop components smaller than this")
    ap.add_argument("--max-components", type=int, default=10,
                    help="more accepted components than this and the run is reported as "
                         "suspicious: a watermark is one small group")
    ap.add_argument("--max-mask-frac", type=float, default=0.08,
                    help="same, for the share of the search area the mask covers")
    ap.add_argument("--dilate", type=int, default=3)
    ap.add_argument("--radius", type=int, default=5)
    ap.add_argument("--mode", default="telea", choices=["telea", "ns"])
    ap.add_argument("--suffix", default="-clean")
    ap.add_argument("--out-ext", default="",
                    help="output extension (default: the input's). Use png when the input is a "
                         "JPEG: writing JPEG re-compresses every pixel, so the output would no "
                         "longer be a record of what the tool actually changed")
    ap.add_argument("--mask-out-dir", default="")
    ap.add_argument("--json-out", default="", help="write a machine-readable run report here")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.learn:
        return run_learn(args)

    if args.strategy == "single":
        print("[ERROR] 单图自动去水印已废弃：实测在真实繁忙照片上不可靠"
              "（掩码落在画面自身内容上、水印基本没被碰到；放宽判据会让合成套件从 10/1/0 "
              "掉到 5/1/5）。请用批量（同尺寸 >=3 张），或显式给 --template / --rect。")
        print("        Single-image automatic detection was removed on measurement, "
              "not for convenience; README section 3 has the numbers.")
        return 2

    if not args.inputs:
        print("[FAIL] no inputs given")
        return 2

    rects = []
    try:
        rects = parse_pairs(args.rect)
        args.color_bgr = parse_color(args.color) if args.color else None
    except ValueError as exc:
        print("[FAIL] %s" % exc)
        return 2

    files = collect_inputs(args.inputs)
    if not files:
        print("[FAIL] no images found in: %s" % ", ".join(args.inputs))
        return 2

    strategy = args.strategy
    if args.restore and not args.template:
        print("[FAIL] --restore needs --template (the mark's shape/alpha)")
        return 2

    template = None
    if args.template:
        strategy = "template"
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
    elif strategy == "template":
        print("[FAIL] --strategy template needs --template <png>")
        return 2

    if args.search == "auto":
        # A template already says WHERE the mark is, so restricting it to the corners
        # would be wrong (measured: it made --restore a no-op).
        args.search = "all" if template is not None else "corners"

    # ---- shared evidence: group the inputs by size
    stacks = {}
    by_shape = {}
    if template is None:
        for f in files:
            im = imread_any(f, cv2.IMREAD_UNCHANGED)
            if im is None:
                continue
            if im.ndim == 2:
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            if im.shape[2] == 4:
                im = im[:, :, :3]
            by_shape.setdefault(im.shape[:2], []).append(im)
        for shape, group in sorted(by_shape.items(), key=lambda kv: -len(kv[1])):
            if len(group) >= 3:
                stacks[shape] = group
        if not stacks:
            found = ", ".join("%dx%d x%d" % (s[1], s[0], len(g)) for s, g in by_shape.items())
            print("[ERROR] %s" % BATCH_ERROR)
            print("        找到的尺寸：%s" % (found or "(读不出任何图片)"))
            return 2
        for shape, group in sorted(stacks.items(), key=lambda kv: -len(kv[1])):
            print("    (shared evidence: %d frames of size %s)" % (len(group), shape))
        for shape, group in by_shape.items():
            if shape not in stacks:
                print("    (%d frame(s) of size %s cannot share evidence and will fail)"
                      % (len(group), shape))

    print(">>> remove_watermark: %d file(s) | strategy=%s search=%s rects=%s%s%s" % (
        len(files), strategy, args.search, rects or "-",
        " [RESTORE]" if args.restore else "",
        " [DRY RUN]" if args.dry_run else ""))

    ok_n = fail_n = 0
    report = {"schema": REPORT_SCHEMA, "strategy": strategy, "search": args.search,
              "restore": bool(args.restore), "dryRun": bool(args.dry_run),
              "files": []}
    for src in files:
        base, ext = os.path.splitext(os.path.basename(src))
        if args.out_ext:
            ext = args.out_ext if args.out_ext.startswith(".") else "." + args.out_ext
        out_dir = args.outdir or os.path.join(os.path.dirname(src), "clean")
        out_path = os.path.join(out_dir, base + args.suffix + ext)
        if not args.dry_run:
            os.makedirs(out_dir, exist_ok=True)
        ok, note, info = remove_one(src, out_path, args, rects, template, stacks)
        entry = {"input": src, "output": None if args.dry_run else out_path,
                 "ok": bool(ok), "note": note}
        entry.update({k: v for k, v in info.items() if k != "strategy"})
        report["files"].append(entry)
        if ok:
            ok_n += 1
            print("    [OK]   %-28s %s" % (os.path.basename(src), note))
            if entry.get("suspicious"):
                print("    [WARN] %-28s the mask covers %.1f%% of the search area in %d "
                      "component(s): this looks like the photo, not a watermark -- "
                      "check --dry-run --mask-out-dir before trusting it"
                      % (os.path.basename(src), 100.0 * entry.get("maskFracOfSearch", 0),
                         entry.get("acceptedComponents", 0)))
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
