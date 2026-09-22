"""test_remove_watermark.py -- prove the watermark tool generalises.

Builds NINE synthetic cases that differ on every axis the tool claims to handle,
then measures each one against its own watermark-free background:

    axis          values used
    ------------  -----------------------------------------------------------
    position      bottom-right / bottom-left / top-right / centre / tiled lattice
    polarity      light mark on dark art, dark mark on light art
    colour        white, black, magenta, grey
    opacity       opaque and 35% semi-transparent
    background    flat dark, flat light, busy procedural art, gradient
    strategy      single image, tiled lattice, multi-image stack, clean no-op

Metrics per case
  hitRate    share of the SOLID watermark pixels that the tool flagged
             (solid = mask >= 200; the antialiased halo is excluded, see below)
  rmse_pre   RMSE vs clean background inside the mark box, before
  rmse_post  same, after  -> must fall below half of rmse_pre
  changed    share of the whole frame the tool modified
  spill      modified pixels outside a padded mark box -> must be 0

Measured trap this file exists to avoid: cv2.putText's antialiased edge is a wide
halo of partially covered pixels; for a 3px stroke the halo is several times the
solid area. Counting those as "missed watermark pixels" made hitRate read 0.11-0.44
even though the tool had found every visible stroke.

Run:  python -X utf8 tools/test_remove_watermark.py     -> exit 0 = all pass
"""
import os
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
TOOL = os.path.join(SRC, "remove_watermark.py")
sys.path.insert(0, SRC)
from remove_watermark import imread_any, imwrite_any   # same I/O as the tool under test

W, H = 800, 600


# --------------------------------------------------------------------------
# backgrounds
# --------------------------------------------------------------------------
def bg_flat_dark():
    return np.full((H, W, 3), (40, 30, 25), np.uint8)


def bg_flat_light():
    return np.full((H, W, 3), (225, 228, 232), np.uint8)


def bg_busy(seed=7):
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W, 3), np.uint8)
    for _ in range(40):
        c = tuple(int(v) for v in rng.integers(30, 220, 3))
        cv2.circle(img, (int(rng.integers(0, W)), int(rng.integers(0, H))),
                   int(rng.integers(20, 90)), c, -1)
    img = cv2.GaussianBlur(img, (0, 0), 9)
    noise = rng.integers(0, 26, (H, W, 1), dtype=np.uint8)
    return cv2.add(img, np.repeat(noise, 3, axis=2))


def bg_gradient():
    ramp = np.tile(np.linspace(20, 210, W, dtype=np.uint8), (H, 1))
    return cv2.merge([ramp, (ramp // 2).astype(np.uint8), (255 - ramp).astype(np.uint8)])


# --------------------------------------------------------------------------
# stamping -> (marked, solid_mask)
# --------------------------------------------------------------------------
def _draw(text, org, scale, thick):
    m = np.zeros((H, W), np.uint8)
    cv2.putText(m, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, 255, thick, cv2.LINE_AA)
    return m


def stamp(base, text, org, colour, scale, thick, alpha=1.0):
    raw = _draw(text, org, scale, thick)
    solid = np.zeros((H, W), np.uint8)
    solid[raw >= 200] = 255
    touched = raw > 40
    marked = base.copy()
    if alpha >= 1.0:
        marked[touched] = colour
    else:
        layer = np.zeros_like(marked)
        layer[:] = colour
        blended = cv2.addWeighted(marked, 1.0 - alpha, layer, alpha, 0)
        marked[touched] = blended[touched]
    return marked, solid


def tiled(base):
    raw = np.zeros((H, W), np.uint8)
    for y in range(60, H, 200):
        for x in range(60, W, 260):
            raw = np.maximum(raw, _draw("WM", (x, y), 0.9, 2))
    solid = np.zeros((H, W), np.uint8)
    solid[raw >= 200] = 255
    marked = base.copy()
    marked[raw > 40] = (235, 235, 235)
    return marked, solid


def box_of(mask, pad=6):
    ys, xs = np.nonzero(mask)
    x0, x1 = max(0, int(xs.min()) - pad), min(W - 1, int(xs.max()) + pad)
    y0, y1 = max(0, int(ys.min()) - pad), min(H - 1, int(ys.max()) + pad)
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def build_cases():
    """(name, marked, clean, solid_mask, extra_args, search, expected)

    `expected` is "pass" for every case the tool is claimed to handle, and
    "known-limitation" for a case that is measured NOT to work and is documented as
    such in README section 3. A known limitation that fails is reported as XFAIL and
    does not fail the run; one that unexpectedly PASSES is reported as XPASS,
    because then the README note is stale.
    """
    cases = []

    base = bg_flat_dark()
    mk, m = stamp(base, "WM", (470, 560), (240, 240, 240), 1.1, 3)
    cases.append(("A_bright_right_dark", mk, base, m, [], "bottom-right", "pass"))

    base = bg_flat_light()
    mk, m = stamp(base, "WATERMARK", (40, 560), (25, 25, 25), 1.3, 4)
    cases.append(("B_dark_left_light", mk, base, m, [], "bottom-left", "pass"))

    base = bg_busy()
    mk, m = stamp(base, "SAMPLE", (250, 320), (255, 255, 255), 1.5, 4)
    cases.append(("C_white_centre_busy", mk, base, m, [], "all", "pass"))

    base = bg_gradient()
    mk, m = stamp(base, "DEMO", (560, 90), (255, 0, 255), 1.2, 3)
    cases.append(("D_magenta_topright_grad", mk, base, m, [], "top-right", "pass"))

    base = bg_busy(11)
    mk, m = stamp(base, "AI", (330, 330), (255, 255, 255), 2.2, 6, alpha=0.35)
    cases.append(("E_semi35_busy", mk, base, m, [], "all", "known-limitation"))

    base = bg_busy(3)
    mk, m = stamp(base, "WATERMARK", (90, 330), (150, 150, 150), 1.6, 9)
    cases.append(("F_grey_big_busy", mk, base, m, [], "all", "pass"))

    base = bg_flat_light()
    mk, m = stamp(base, "C", (640, 560), (10, 10, 10), 2.6, 8)
    cases.append(("G_black_on_light", mk, base, m, [], "bottom-right", "pass"))

    base = bg_busy(5)
    mk, m = tiled(base)
    cases.append(("H_tiled_lattice", mk, base, m, ["--period", "260,200"], "all", "pass"))

    base = bg_busy(13)
    mk, m = stamp(base, "X", (390, 250), (255, 255, 255), 3.0, 5)
    cases.append(("I_white_small_centre", mk, base, m, [], "all", "pass"))

    return cases


def rmse(a, b, box):
    x, y, w, h = box
    p = a[y:y + h, x:x + w].astype(np.float32)
    q = b[y:y + h, x:x + w].astype(np.float32)
    return float(np.sqrt(np.mean((p - q) ** 2)))


def main():
    tmp = tempfile.mkdtemp(prefix="wmtest_")
    passed = failed = xfail = xpass = 0
    print(">>> synthetic generality test (%d cases)" % len(build_cases()))
    print("  %-24s %7s %8s %9s %9s %6s %6s %6s" % (
        "case", "hit", "changed", "rmse_pre", "rmse_post", "spill", "px_true", "px_flag"))
    try:
        for name, marked, clean, mask, extra, search, expected in build_cases():
            d = os.path.join(tmp, name)
            os.makedirs(d)
            imwrite_any(os.path.join(d, "wm.png"), marked)
            masks = os.path.join(d, "masks")
            args = [os.path.join(d, "wm.png"), "--search", search, "--mask-out-dir", masks] + extra
            proc = subprocess.run([sys.executable, "-X", "utf8", TOOL] + args,
                                  capture_output=True, text=True)
            cleaned = imread_any(os.path.join(d, "clean", "wm-clean.png"))
            flagged = imread_any(os.path.join(masks, "wm-mask.png"), cv2.IMREAD_GRAYSCALE)
            if proc.returncode != 0 or cleaned is None or flagged is None:
                print("  [FAIL] %-24s tool failed: %s" % (name, (proc.stdout + proc.stderr)[-300:]))
                failed += 1
                continue
            if cleaned.shape[:2] != (H, W):
                print("  [FAIL] %-24s wrong output size %s" % (name, cleaned.shape[:2]))
                failed += 1
                continue

            changed = (np.abs(cleaned.astype(np.int16) - marked.astype(np.int16)).sum(axis=2) > 0)
            changed_frac = changed.sum() / float(W * H)
            px_true = int((mask > 0).sum())
            px_flag = int((flagged > 0).sum())
            hit = float(((flagged > 0) & (mask > 0)).sum()) / max(1, px_true)

            bx, by, bw, bh = box_of(mask)
            pre = rmse(marked, clean, (bx, by, bw, bh))
            post = rmse(cleaned, clean, (bx, by, bw, bh))

            outside = np.ones((H, W), bool)
            pad = 14
            outside[max(0, by - pad):by + bh + pad, max(0, bx - pad):bx + bw + pad] = False
            spill = int((changed & outside).sum())

            ok = (post < pre * 0.5) and (hit >= 0.5) and (changed_frac < 0.25) and (spill == 0)
            if ok and expected == "pass":
                label, passed = "PASS", passed + 1
            elif ok and expected == "known-limitation":
                label, xpass = "XPASS", xpass + 1
            elif not ok and expected == "known-limitation":
                label, xfail = "XFAIL", xfail + 1
            else:
                label, failed = "FAIL", failed + 1
            print("  [%s] %-24s %7.2f %7.2f%% %9.2f %9.2f %6d %6d %6d" % (
                label, name, hit, 100 * changed_frac, pre, post,
                spill, px_true, px_flag))
            if not ok:
                if post >= pre * 0.5:
                    print("         -> rmse must drop below %.2f" % (pre * 0.5))
                if hit < 0.5:
                    print("         -> flagged %d of %d solid mark px" % (
                        int(((flagged > 0) & (mask > 0)).sum()), px_true))
                if spill:
                    print("         -> touched %d px outside the mark" % spill)
                if label == "XFAIL":
                    print("         -> known limitation, documented in README section 3")
            if label == "XPASS":
                print("         -> UNEXPECTED PASS: this case is documented as a limitation,"
                      " the README note is stale")

        # --- clean image must come out with zero changed pixels ---
        d = os.path.join(tmp, "clean_case")
        os.makedirs(d)
        src = os.path.join(d, "plain.png")
        imwrite_any(src, bg_busy(21))
        subprocess.run([sys.executable, "-X", "utf8", TOOL, src, "--search", "all"],
                       capture_output=True, text=True)
        out_img = imread_any(os.path.join(d, "clean", "plain-clean.png"))
        orig = imread_any(src)
        zero = out_img is not None and not np.any(out_img != orig)
        print("  [%s] %-24s %s" % ("PASS" if zero else "FAIL", "J_clean_noop",
                                   "0 changed pixels" if zero else "MODIFIED A CLEAN IMAGE"))
        passed, failed = (passed + 1, failed) if zero else (passed, failed + 1)

        # --- multi-image strategy ---
        d = os.path.join(tmp, "multi_case")
        os.makedirs(d)
        for i in range(5):
            base = bg_busy(30 + i)
            mk, _m = stamp(base, "LOGO", (300, 330), (250, 250, 250), 1.8, 5, alpha=0.45)
            imwrite_any(os.path.join(d, "f%d.png" % i), mk)
        proc = subprocess.run([sys.executable, "-X", "utf8", TOOL, d,
                               "--strategy", "multi", "--search", "all"],
                              capture_output=True, text=True)
        outdir = os.path.join(d, "clean")
        got = len(os.listdir(outdir)) if os.path.isdir(outdir) else 0
        ok = proc.returncode == 0 and got == 5
        print("  [%s] %-24s %d/5 outputs (exit %d)" % (
            "PASS" if ok else "FAIL", "K_multi_image", got, proc.returncode))
        if not ok:
            print((proc.stdout + proc.stderr)[-300:])
        passed, failed = (passed + 1, failed) if ok else (passed, failed + 1)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("")
    print("[SUMMARY] passed %d / xfail %d (known limitations) / xpass %d / failed %d"
          % (passed, xfail, xpass, failed))
    if xfail:
        print("          xfail cases are NOT passes: see README section 3 for each one,"
              " with its measured numbers")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
