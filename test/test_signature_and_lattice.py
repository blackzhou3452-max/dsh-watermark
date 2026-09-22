"""test_signature_and_lattice.py -- strategy A (exact signature) and the lattice.

The generality suite (test_remove_watermark.py) covers the heuristic detector.
This file covers the machinery that was NOT in the first version, and the one
bug the old README documented but never fixed:

  L1  --learn      solve alpha and colour from a marked/clean pair, and check the
                   solved values against the values that were used to build the
                   pair (an absolute check, not "it ran")
  L2  --restore    invert the overlay exactly and measure how much of the mark's
                   damage it undoes
  L3  negative     pixels the signature does not cover must come out BIT-IDENTICAL
                   (this is the promise the whole tool rests on)
  L4  negative     a pair with no mark must be refused, not turned into a signature
  P1  --period auto  the lattice is found from the image, and the tiled mask built
                   from it is measured with the same hit/spill metrics as the suite
  B1  regression   the big semi-transparent mark that the old detector scored
                   0 pixels on (README section 3, first warning)

Run:  python -X utf8 test/test_signature_and_lattice.py     -> exit 0 = all pass
"""
import json
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
sys.path.insert(0, HERE)
from remove_watermark import imread_any, imwrite_any                      # noqa: E402
from test_remove_watermark import (bg_busy, bg_flat_dark, bg_gradient,     # noqa: E402
                                   box_of, stamp, tiled, W, H)

RESULTS = []


def run(args):
    return subprocess.run([sys.executable, "-X", "utf8", TOOL] + args,
                          capture_output=True, text=True)


def rmse(a, b, box=None):
    if box is None:
        p, q = a.astype(np.float32), b.astype(np.float32)
    else:
        x, y, w, h = box
        p = a[y:y + h, x:x + w].astype(np.float32)
        q = b[y:y + h, x:x + w].astype(np.float32)
    return float(np.sqrt(np.mean((p - q) ** 2)))


def record(name, ok, detail):
    RESULTS.append((name, ok, detail))
    print("  [%s] %-22s %s" % ("PASS" if ok else "FAIL", name, detail))


def case_learn_and_restore(tmp):
    """L1/L2/L3: solve a known mark from a pair, reuse it, invert it exactly."""
    true_alpha, true_colour = 0.40, (255, 255, 255)
    base = bg_gradient()
    marked, solid = stamp(base, "WATERMARK", (120, 420), true_colour, 2.4, 7,
                          alpha=true_alpha)
    d = os.path.join(tmp, "learn")
    os.makedirs(d)
    wm = os.path.join(d, "shot_wm.png")
    cl = os.path.join(d, "shot_clean.png")
    imwrite_any(wm, marked)
    imwrite_any(cl, base)
    sig = os.path.join(d, "learned.png")
    proc = run(["--learn", wm, cl, "--signature-out", sig])
    if proc.returncode != 0:
        record("L1_learn_solves", False, "tool failed: %s" % (proc.stdout + proc.stderr)[-200:])
        return
    meta_path = os.path.splitext(sig)[0] + ".json"
    meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
    a_peak = float(meta.get("alphaPeak", 0.0))
    colour = meta.get("colorBGR", [0, 0, 0])
    # the pair was built with a uniform alpha, so the learned peak must be that alpha
    ok_a = abs(a_peak - true_alpha) <= 0.03
    ok_c = all(abs(c - 255.0) <= 3.0 for c in colour)
    explained = meta.get("pairs", [{}])[0].get("explained")
    record("L1_learn_solves", ok_a and ok_c,
           "alpha %.3f (built %.2f, tol .03) colourBGR %s (built white) explained %s"
           % (a_peak, true_alpha, [round(c, 1) for c in colour], explained))

    # L2: restore the marked image with the learned signature
    outdir = os.path.join(d, "out")
    proc = run([wm, "--template", sig, "--restore", "-o", outdir])
    restored = imread_any(os.path.join(outdir, "shot_wm-clean.png"))
    if proc.returncode != 0 or restored is None:
        record("L2_restore_inverts", False, "tool failed: %s" % (proc.stdout + proc.stderr)[-200:])
        return
    box = box_of(solid)
    pre = rmse(marked, base, box)
    post = rmse(restored, base, box)
    record("L2_restore_inverts", post < pre * 0.35,
           "rmse vs the true clean image inside the mark: %.2f -> %.2f (needs < %.2f)"
           % (pre, post, pre * 0.35))

    # L3: outside the signature's support nothing may change, bit for bit
    alpha_png = imread_any(sig, cv2.IMREAD_UNCHANGED)
    support = np.zeros(marked.shape[:2], np.uint8)
    if alpha_png.ndim == 3 and alpha_png.shape[2] == 4:
        x, y, bw, bh = meta["bbox"]
        support[y:y + alpha_png.shape[0], x:x + alpha_png.shape[1]] = alpha_png[:, :, 3]
    else:
        support[:] = 255
    outside = support == 0
    changed_outside = int(np.any(restored[outside] != marked[outside], axis=-1).sum()) \
        if outside.any() else 0
    record("L3_negative_untouched", changed_outside == 0,
           "%d px outside the signature support, %d of them changed (must be 0)"
           % (int(outside.sum()), changed_outside))


def case_learn_refuses_empty(tmp):
    """L4: identical pair (no mark) must be refused with exit 2, not 'solved'."""
    d = os.path.join(tmp, "nopair")
    os.makedirs(d)
    img = bg_busy(4)
    a = os.path.join(d, "a.png")
    b = os.path.join(d, "b.png")
    imwrite_any(a, img)
    imwrite_any(b, img)
    sig = os.path.join(d, "should-not-exist.png")
    proc = run(["--learn", a, b, "--signature-out", sig])
    refused = proc.returncode == 2 and not os.path.exists(sig)
    record("L4_learn_refuses_empty", refused,
           "exit %d, signature written: %s, message: %s"
           % (proc.returncode, os.path.exists(sig),
              (proc.stdout + proc.stderr).strip().splitlines()[-1][:90] if
              (proc.stdout + proc.stderr).strip() else "(none)"))


def case_period_auto(tmp):
    """P1: the tiled lattice is estimated from the image and used."""
    base = bg_busy(5)
    marked, solid = tiled(base)          # true lattice: x step 260, y step 200
    d = os.path.join(tmp, "lattice")
    os.makedirs(d)
    src = os.path.join(d, "wm.png")
    imwrite_any(src, marked)
    masks = os.path.join(d, "masks")
    proc = run([src, "--search", "all", "--period", "auto", "--mask-out-dir", masks])
    line = [ln for ln in proc.stdout.splitlines() if "auto period" in ln]
    changed = None
    flagged = imread_any(os.path.join(masks, "wm-mask.png"), cv2.IMREAD_GRAYSCALE)
    if proc.returncode == 0 and flagged is not None:
        cleaned = imread_any(os.path.join(d, "clean", "wm-clean.png"))
        changed = (np.abs(cleaned.astype(np.int16) - marked.astype(np.int16)).sum(axis=2) > 0)
    hit = float(((flagged > 0) & (solid > 0)).sum()) / max(1, int((solid > 0).sum())) \
        if flagged is not None else 0.0
    outside = np.ones((H, W), bool)
    bx, by, bw, bh = box_of(solid)
    outside[max(0, by - 14):by + bh + 14, max(0, bx - 14):bx + bw + 14] = False
    spill = int((changed & outside).sum()) if changed is not None else -1
    ok = proc.returncode == 0 and hit >= 0.5 and spill == 0
    record("P1_period_auto", ok,
           "hit %.2f, spill %d, exit %d | %s"
           % (hit, spill, proc.returncode, (line[0].strip() if line else "(no estimate line)")))


def case_big_semi_transparent(tmp):
    """B1: the big semi-transparent mark the OLD detector scored 0 px on.

    README section 3, first warning: "a big magenta watermark, radius 31 window,
    detected 0 pixels". Reproduced here at the same scale, as a regression case.
    """
    base = bg_busy(17)
    marked, solid = stamp(base, "DEMO", (150, 400), (255, 0, 255), 3.6, 11, alpha=0.5)
    d = os.path.join(tmp, "bigmark")
    os.makedirs(d)
    src = os.path.join(d, "wm.png")
    imwrite_any(src, marked)
    masks = os.path.join(d, "masks")
    proc = run([src, "--search", "all", "--mask-out-dir", masks])
    flagged = imread_any(os.path.join(masks, "wm-mask.png"), cv2.IMREAD_GRAYSCALE)
    if proc.returncode != 0 or flagged is None:
        record("B1_big_semi_transparent", False,
               "tool failed: %s" % (proc.stdout + proc.stderr)[-200:])
        return
    hit = float(((flagged > 0) & (solid > 0)).sum()) / max(1, int((solid > 0).sum()))
    record("B1_big_semi_transparent", hit >= 0.5,
           "hit %.2f on a %d px half-transparent mark (the old detector: 0 flagged px)"
           % (hit, int((solid > 0).sum())))

    # and the same image WITHOUT the mark must not be touched
    plain = os.path.join(d, "plain.png")
    imwrite_any(plain, base)
    run([plain, "--search", "all"])
    out = imread_any(os.path.join(d, "clean", "plain-clean.png"))
    zero = out is not None and not np.any(out != base)
    record("B2_negative_untouched", zero,
           "the same busy frame without the mark: %s"
           % ("0 changed pixels" if zero else "MODIFIED"))


def main():
    tmp = tempfile.mkdtemp(prefix="wmsig_")
    print(">>> signature / lattice / regression test")
    try:
        case_learn_and_restore(tmp)
        case_learn_refuses_empty(tmp)
        case_period_auto(tmp)
        case_big_semi_transparent(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    passed = sum(1 for _n, ok, _d in RESULTS if ok)
    failed = len(RESULTS) - passed
    print("")
    print("[SUMMARY] passed %d / failed %d" % (passed, failed))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
