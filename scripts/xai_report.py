#!/usr/bin/env python
"""Turn a run_xai_all.sh output directory into one readable results report.

    python scripts/xai_report.py results/eval/xai/<stamp> [-o RESULTS.txt]

Every number the report needs comes out of the per-run results_by_threshold.pkl
files, joined to each model's grid manifest.json on the grid filename so the
scores can be split by source dataset (the manifest's 'pool' field).

Sections:
  1. run configuration (selection protocol, seed, grids per dataset)
  2. grid inventory      -- how many grids each dataset actually contributed
  3. headline table      -- every model x method at the reporting thresholds
  4. threshold sweep     -- full curve per model x method, weighted + unweighted
  5. per-dataset tables  -- one block per reporting threshold
"""
import argparse
import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

CHANCE = 1.0 / 9.0
# 0 = raw map, 0.1 = noise floor removed, 0.5 = mid sweep, 1.0 = argmax pixel
# (the classic pointing game), top-k = threshold-free and the fairest single
# number to headline. See the notes at the end of the report.
#
# Read threshold 0 with care: unweighted@0 is exactly 1/9 by construction for
# any CAM (after ReLU + max-normalisation every pixel is > 0, so "active
# fraction in the fake cell" measures nothing), and weighted@0 is dominated by
# a method's diffuse positive floor rather than by where its peak sits.
REPORT_THRESHOLDS = [0, 0.1, 0.5, 1.0, "top0.025"]


def thr_key(t):
    return (1, str(t)) if isinstance(t, str) else (0, t)


def find_results(model_dir):
    """{method: results_by_threshold} for one model directory."""
    out = {}
    if not os.path.isdir(model_dir):
        return out
    for method in sorted(os.listdir(model_dir)):
        mdir = os.path.join(model_dir, method)
        if not os.path.isdir(mdir) or method.startswith("grids"):
            continue
        for dirpath, _, files in os.walk(mdir):
            if "results_by_threshold.pkl" in files:
                with open(os.path.join(dirpath, "results_by_threshold.pkl"), "rb") as f:
                    out[method] = pickle.load(f)
                break
    return out


def load_manifest(model_dir):
    path = os.path.join(model_dir, "grids", "3x3", "manifest.json")
    if not os.path.exists(path):
        return {}, {}
    man = json.load(open(path))
    f2d = {g["file"]: (g.get("pool") if g.get("pool") != "__all__"
                       else g.get("fake_dataset", "?"))
           for g in man["grids"]}
    meta = {k: man.get(k) for k in ("selection", "real_selection", "seed", "note")}
    return f2d, meta


def collect(run_dir):
    runs, manifests, meta = {}, {}, {}
    for model in sorted(os.listdir(run_dir)):
        mdir = os.path.join(run_dir, model)
        if not os.path.isdir(mdir) or model == "logs":
            continue
        res = find_results(mdir)
        if not res:
            continue
        manifests[model], m = load_manifest(mdir)
        meta.update({k: v for k, v in m.items() if v is not None})
        for method, raw in res.items():
            runs[(model, method)] = raw
    return runs, manifests, meta


def mean_at(entries, weighted=True):
    key = "weighted_localization_score" if weighted else "unweighted_localization_score"
    return float(np.mean([e[key] for e in entries]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("-o", "--out", default=None,
                    help="output file (default: <run_dir>/RESULTS.txt)")
    args = ap.parse_args()

    runs, manifests, meta = collect(args.run_dir)
    if not runs:
        sys.exit(f"no finished results under {args.run_dir}")

    out_path = args.out or os.path.join(args.run_dir, "RESULTS.txt")
    lines = []
    def w(s=""):
        lines.append(s)

    cols = sorted(runs)
    label = {c: f"{c[0]}/{c[1]}" for c in cols}
    width = max(22, max(len(v) for v in label.values()) + 2)

    w("=" * 100)
    w("GRID POINTING GAME -- RESULTS")
    w("=" * 100)
    w(f"run directory : {args.run_dir}")
    w(f"chance level  : {CHANCE:.4f}  (1 fake cell out of 3x3)")
    for k in ("selection", "real_selection", "seed", "note"):
        if meta.get(k) is not None:
            w(f"{k:<14}: {meta[k]}")
    w(f"models x methods : {len(cols)}")
    w("")

    # ---- 2. grid inventory --------------------------------------------------
    w("-" * 100)
    w("GRID INVENTORY  (grids actually built per dataset; a short count means the")
    w("model's pool of confidently-correct images ran out, not a failure)")
    w("-" * 100)
    for model in sorted(manifests):
        counts = defaultdict(int)
        for ds in manifests[model].values():
            counts[ds] += 1
        total = sum(counts.values())
        w(f"{model}  (total {total})")
        for ds in sorted(counts):
            w(f"    {ds:<20} {counts[ds]:>5}")
        w("")

    # ---- 3. headline table --------------------------------------------------
    thr_present = [t for t in REPORT_THRESHOLDS
                   if any(t in runs[c] for c in cols)]
    w("-" * 100)
    w("HEADLINE  --  mean weighted localization score")
    w("-" * 100)
    w(f"{'model / method':<{width}}" + "".join(f"{str(t):>12}" for t in thr_present)
      + f"{'n grids':>10}")
    for c in cols:
        row = "".join(f"{mean_at(runs[c][t]):>12.4f}" if t in runs[c] else f"{'-':>12}"
                      for t in thr_present)
        n = len(runs[c][sorted(runs[c], key=thr_key)[0]])
        w(f"{label[c]:<{width}}{row}{n:>10}")
    w("")

    # ---- 4. full threshold sweep -------------------------------------------
    w("-" * 100)
    w("THRESHOLD SWEEP  --  weighted / unweighted")
    w("-" * 100)
    for c in cols:
        w(f"{label[c]}")
        w(f"    {'threshold':>10} {'weighted':>10} {'unweighted':>12} {'n':>7}")
        for t in sorted(runs[c], key=thr_key):
            e = runs[c][t]
            w(f"    {str(t):>10} {mean_at(e):>10.4f} {mean_at(e, False):>12.4f} {len(e):>7}")
        w("")

    # ---- 5. per-dataset -----------------------------------------------------
    for t in thr_present:
        per = {}
        for c in cols:
            if t not in runs[c]:
                continue
            f2d = manifests.get(c[0], {})
            by = defaultdict(list)
            for e in runs[c][t]:
                by[f2d.get(os.path.basename(e["path"]), "?")].append(
                    e["weighted_localization_score"])
            per[c] = {k: (float(np.mean(v)), len(v)) for k, v in by.items()}
        if not per:
            continue
        datasets = sorted({d for v in per.values() for d in v})
        w("-" * 100)
        w(f"PER-DATASET  --  weighted localization score @ threshold {t}")
        w("-" * 100)
        w(f"{'dataset':<20}" + "".join(f"{label[c]:>{width}}" for c in per))
        for ds in datasets:
            row = ""
            for c in per:
                v = per[c].get(ds)
                row += f"{v[0]:>{width}.4f}" if v else f"{'-':>{width}}"
            w(f"{ds:<20}{row}")
        w(f"{'(n grids)':<20}" + "".join(
            f"{max((v[1] for v in per[c].values()), default=0):>{width}d}" for c in per))
        w("")

    # ---- reading notes ------------------------------------------------------
    w("-" * 100)
    w("HOW TO READ THESE NUMBERS")
    w("-" * 100)
    w(f"chance = {CHANCE:.4f}. Both scores measure how much of the explanation landed")
    w("in the cell holding the fake.")
    w("  weighted    fraction of positive attribution MASS in the fake cell.")
    w("  unweighted  fraction of ACTIVE PIXELS in the fake cell after thresholding.")
    w("  threshold t map is normalised to [0,1] and everything below t is zeroed")
    w("              before scoring. t=1.0 is effectively the classic pointing game.")
    w("  top0.025    keeps only the strongest 2.5% of pixels -- threshold-free, so it")
    w("              neither rewards nor punishes a method for its output scaling.")
    w("")
    w("CAVEATS")
    w("  * unweighted @ threshold 0 is meaningless for CAM methods: after ReLU and")
    w("    max-normalisation every pixel is > 0, so the score is exactly 1/9 by")
    w("    construction and measures nothing.")
    w("  * weighted @ threshold 0 conflates map concentration with accuracy -- a")
    w("    correct peak sitting on a broad positive floor scores low. This is why the")
    w("    ranking of methods can reverse between low and high thresholds.")
    w("  * top0.025 is the fairest single number to headline; quote the sweep with it.")
    w("  * scores at or near 1.0000 mean the protocol SATURATED and has no")
    w("    discriminating power left -- do not read a winner out of such a column.")
    w("  * with confidence selection each model gets its OWN grids, so the METHOD")
    w("    comparison is matched but the ARCHITECTURE comparison is not. Use")
    w("    --shared-assets for matched inputs across models.")
    w("  * a dataset with few grids has a correspondingly noisy mean; see the")
    w("    inventory above.")

    text = "\n".join(lines) + "\n"
    with open(out_path, "w") as f:
        f.write(text)
    print(text)
    print(f"[written to {out_path}]", file=sys.stderr)


if __name__ == "__main__":
    main()
