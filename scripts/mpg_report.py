#!/usr/bin/env python
"""Turn a run_mpg_all.sh output directory into one readable results report.

    python scripts/mpg_report.py results/eval/mpg/<stamp> [-o RESULTS.txt]

run_mpg_all.sh writes one MPG_eval run per (model, method, dataset) into
    <run>/<model>/<method>/<dataset>/<model_name>_<config>/MaskPointingGame/
so model, method and dataset are recovered from the path -- there is no grid
manifest to join against (the MPG scores single images, not grids).
"""
import argparse
import glob
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

REPORT_THRESHOLDS = [0, 0.1, 0.5, 1.0, "top0.025"]


def thr_key(t):
    return (1, str(t)) if isinstance(t, str) else (0, t)


def collect(run_dir):
    """{(model, method, dataset): results_by_threshold}"""
    runs = {}
    pattern = os.path.join(run_dir, "*", "*", "*", "*", "MaskPointingGame",
                           "results_by_threshold.pkl")
    for path in sorted(glob.glob(pattern)):
        rel = os.path.relpath(path, run_dir).split(os.sep)
        model, method, dataset = rel[0], rel[1], rel[2]
        with open(path, "rb") as f:
            runs[(model, method, dataset)] = pickle.load(f)
    return runs


def mean_at(entries, weighted=True):
    key = "weighted_localization_score" if weighted else "unweighted_localization_score"
    vals = [e[key] for e in entries if key in e]
    return float(np.mean(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    runs = collect(args.run_dir)
    if not runs:
        sys.exit(f"no finished MPG results under {args.run_dir}")

    lines = []
    def w(s=""):
        lines.append(s)

    models = sorted({k[0] for k in runs})
    methods = sorted({k[1] for k in runs})
    datasets = sorted({k[2] for k in runs})
    pairs = sorted({(k[0], k[1]) for k in runs})
    label = {p: f"{p[0]}/{p[1]}" for p in pairs}
    width = max(22, max(len(v) for v in label.values()) + 2)

    w("=" * 100)
    w("MASK POINTING GAME -- RESULTS")
    w("=" * 100)
    w(f"run directory : {args.run_dir}")
    w(f"models        : {len(models)}   methods: {len(methods)}   datasets: {len(datasets)}")
    w(f"datasets      : {', '.join(datasets)}")
    w("")

    # ---- inventory ----------------------------------------------------------
    w("-" * 100)
    w("INVENTORY  (images scored per model/method/dataset)")
    w("-" * 100)
    w(f"{'model / method':<{width}}" + "".join(f"{d:>16}" for d in datasets))
    for p in pairs:
        row = ""
        for d in datasets:
            r = runs.get((p[0], p[1], d))
            n = len(r[sorted(r, key=thr_key)[0]]) if r else 0
            row += f"{n:>16}" if r else f"{'-':>16}"
        w(f"{label[p]:<{width}}{row}")
    w("")

    # ---- headline: pooled over datasets ------------------------------------
    thr_present = sorted({t for r in runs.values() for t in r if t in REPORT_THRESHOLDS},
                         key=thr_key)
    w("-" * 100)
    w("HEADLINE  --  mean weighted localization score, pooled over all datasets")
    w("-" * 100)
    w(f"{'model / method':<{width}}" + "".join(f"{str(t):>12}" for t in thr_present)
      + f"{'n images':>10}")
    for p in pairs:
        merged = defaultdict(list)
        for d in datasets:
            r = runs.get((p[0], p[1], d))
            if not r:
                continue
            for t, entries in r.items():
                merged[t].extend(entries)
        row = "".join(f"{mean_at(merged[t]):>12.4f}" if t in merged else f"{'-':>12}"
                      for t in thr_present)
        n = len(merged[thr_present[0]]) if thr_present[0] in merged else 0
        w(f"{label[p]:<{width}}{row}{n:>10}")
    w("")

    # ---- full sweep ---------------------------------------------------------
    w("-" * 100)
    w("THRESHOLD SWEEP  --  weighted / unweighted, pooled over datasets")
    w("-" * 100)
    for p in pairs:
        merged = defaultdict(list)
        for d in datasets:
            r = runs.get((p[0], p[1], d))
            if r:
                for t, entries in r.items():
                    merged[t].extend(entries)
        w(f"{label[p]}")
        w(f"    {'threshold':>10} {'weighted':>10} {'unweighted':>12} {'n':>7}")
        for t in sorted(merged, key=thr_key):
            w(f"    {str(t):>10} {mean_at(merged[t]):>10.4f} "
              f"{mean_at(merged[t], False):>12.4f} {len(merged[t]):>7}")
        w("")

    # ---- per dataset --------------------------------------------------------
    for t in thr_present:
        w("-" * 100)
        w(f"PER-DATASET  --  weighted localization score @ threshold {t}")
        w("-" * 100)
        w(f"{'model / method':<{width}}" + "".join(f"{d:>16}" for d in datasets))
        for p in pairs:
            row = ""
            for d in datasets:
                r = runs.get((p[0], p[1], d))
                row += f"{mean_at(r[t]):>16.4f}" if (r and t in r) else f"{'-':>16}"
            w(f"{label[p]:<{width}}{row}")
        w("")

    # ---- reading notes ------------------------------------------------------
    w("-" * 100)
    w("HOW TO READ THESE NUMBERS")
    w("-" * 100)
    w("The MPG scores how much of the explanation falls inside the ground-truth")
    w("manipulation mask of a single fake image.")
    w("  weighted    fraction of positive attribution MASS inside the mask.")
    w("  unweighted  fraction of ACTIVE PIXELS inside the mask after thresholding.")
    w("")
    w("THERE IS NO FIXED CHANCE LEVEL, unlike the GPG's 1/9. A uniform map scores")
    w("mask_area / image_area, which varies per image -- so compare methods within")
    w("a dataset, never a raw MPG number against a raw GPG number.")
    w("")
    w("CAVEATS")
    w("  * MPG-weighted REWARDS DIFFUSE MAPS. The FF++ mask covers most of the face")
    w("    and the face fills most of a 256px aligned crop, so a map spread over the")
    w("    whole image captures most of the mask mass. Measured on 500 shared FF++")
    w("    images, this REVERSED the GPG ordering: gradcam-on-b-cos scored WORST on")
    w("    GPG (0.318) and BEST on MPG (0.979). Do not headline weighted MPG.")
    w("  * the unweighted@0 row is close to 'what fraction of the image is active',")
    w("    so an ordering that is the reverse of the weighted one is the signature")
    w("    of that coverage artifact rather than a localization difference.")
    w("  * to make MPG discriminative, normalise by mask area (use mask_area /")
    w("    image_area as the per-image baseline) or restrict to small-mask images.")
    w("    Neither is implemented yet.")
    w("  * only the FaceForensics++ family has masks, so MPG says nothing about")
    w("    cross-dataset or fully-synthetic generalisation -- that is the GPG's job.")
    w("  * the image list is model-free and shared, so unlike the confidence-selected")
    w("    GPG, BOTH the method and the architecture axis are matched here.")

    out_path = args.out or os.path.join(args.run_dir, "RESULTS.txt")
    text = "\n".join(lines) + "\n"
    with open(out_path, "w") as f:
        f.write(text)
    print(text)
    print(f"[written to {out_path}]", file=sys.stderr)


if __name__ == "__main__":
    main()
