#!/usr/bin/env bash
# Run the Grid Pointing Game for every trained detector x every XAI method.
#
#   scripts/run_xai_all.sh                          # all families, default datasets
#   scripts/run_xai_all.sh --family resnet          # only the resnet34 pair
#   scripts/run_xai_all.sh --family "vit convnext"  # two families, one argument
#   scripts/run_xai_all.sh --datasets "FaceForensics++ Celeb-DF-v2"
#   scripts/run_xai_all.sh --shared-assets          # fixed shared random grids instead
#   scripts/run_xai_all.sh --grids-per-dataset 50   # default 100
#   scripts/run_xai_all.sh --methods "bcos gradcam" # default: all six
#   scripts/run_xai_all.sh --real-selection confident   # default: shuffle
#   scripts/run_xai_all.sh --frame-num 4            # default 32 (= all stored frames)
#   scripts/run_xai_all.sh --keep-grids             # keep grid tensors (~10 GB/model)
#   scripts/run_xai_all.sh --include-b1             # b=1 configs are skipped by default
#   scripts/run_xai_all.sh --dry-run                # print what would run
#
# Results land in <out>/RESULTS.txt -- one readable file with the grid inventory,
# the headline table, the full threshold sweep and per-dataset breakdowns.
# Regenerate it any time with: python scripts/xai_report.py <out>
#
# Families: resnet, xception, vit, convnext (default: all) -- each covers the
# standard model and its whole b-cos b-value sweep. Checkpoint discovery is
# identical to test_all_detectors.sh (run dir names are inconsistent).
#
# DEFAULT PROTOCOL (--shared-assets off): the B-cos-v2 mirror. Grids are built
# per model from its OWN most-confident correctly-classified images, ranked by
# raw max logit, with every cell of a grid drawn from ONE dataset
# (--dataset-mixing single) so no cell is identifiable by domain signature.
# Grids are built ONCE per model and reused by every XAI method, so the method
# comparison is exactly matched; the models see DIFFERENT grids, so the
# architecture axis is not. Use --shared-assets for the matched-input variant.
#
# Every run sweeps thresholds 0.0..1.0 (11 steps) plus the top-k mass variant,
# and the per-run mean scores are appended to its log and to the final summary.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

PYTHON="${PYTHON:-$HOME/miniconda3/envs/bcos/bin/python}"
GPG="notebooks/Linus/GridPointingGame/GPG_eval.py"
TEST_CFG="training/config/test_config.yaml"

CKPT_SPLIT="val"
DRY_RUN=0
INCLUDE_INCOMPLETE=0
INCLUDE_B1=0        # b=1 models are linear and never work -- see the skip below
# A 3x3 grid of 256px cells is 768x768x3 float32 = 7.1 MB. At ~1400 grids that
# is ~10 GB PER MODEL, ~220 GB for a full sweep, for tensors only that model's
# own method runs ever read. They are deleted once its methods finish; the
# manifest keeps every source image, cell position and seed, so any grid is
# exactly reproducible. --keep-grids to retain them.
KEEP_GRIDS=0
FAMILIES=""
GRIDS_PER_DATASET=100
THRESHOLD_STEPS=10
SEED=32
# Preprocessing wrote exactly 32 frames per video, so 32 = use them all; lower
# values evenly subsample. Kept at the ceiling because the real pools need it:
# at 4 frames/video most datasets cannot supply the 800 reals a 100-grid run
# needs. Image-only sets (MidJourney/whichisreal/CollabDiff) hold 1 frame per
# entry and ignore this entirely.
FRAME_NUM=32
# 'shuffle' = uniform random draw from the dataset's confident reals;
# 'confident' = most-confident-first (what B-cos-v2 does). Shuffle matters at
# FRAME_NUM=32: confidence is ranked globally, so most-confident-first tends to
# pick 8 consecutive frames of ONE video, giving a grid of near-identical crops.
# The shuffle is effectively per dataset -- GPG_eval shuffles the global list and
# then partitions it into per-dataset pools stably, so each pool stays uniformly
# randomly ordered, and --dataset-mixing single keeps cells within one dataset.
REAL_SELECTION="shuffle"
SHARED_ASSETS=""     # non-empty => evaluate this shared grid folder, no ranking
SHARED_DEFAULT="results/GPG_assets/shared_random/FaceForensics++_test_256/3x3"
RUNNING_THRESHOLD_MIN=30

# 'bcos' is appended automatically for b-cos configs (it needs a b-cos model).
METHODS_DEFAULT="gradcam xgrad grad++ layergrad ig lime"
METHODS=""

# run them later: FSAll_cdf FRAll_cdf EFSAll_cdf
DATASETS_DEFAULT="FaceForensics++ Celeb-DF-v1 Celeb-DF-v2 DFDCP DFDC UADFV FSAll_ff FRAll_ff EFSAll_ff deepfacelab heygen MidJourney whichisreal CollabDiff e4e_ff e4e_cdf"
DATASETS="$DATASETS_DEFAULT"

df40_datasets() {
    ls preprocessing/dataset_json_v3/*.json \
        | xargs -n1 basename | sed 's/\.json$//' \
        | grep -vE '^(FaceForensics\+\+|FF-(DF|F2F|FS|NT)|Celeb-DF-v[12]|DFDCP?|UADFV|DF40_all|stargan|starganv2|styleclip)$' \
        | tr '\n' ' '
}

usage() {
    sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --df40)                DATASETS="$(df40_datasets)"; shift ;;
        --datasets)            DATASETS="$2"; shift 2 ;;
        --family|--families)   FAMILIES="$FAMILIES ${2//,/ }"; shift 2 ;;
        --methods)             METHODS="$METHODS ${2//,/ }"; shift 2 ;;
        --grids-per-dataset)   GRIDS_PER_DATASET="$2"; shift 2 ;;
        --threshold-steps)     THRESHOLD_STEPS="$2"; shift 2 ;;
        --frame-num)           FRAME_NUM="$2"; shift 2 ;;
        --real-selection)      REAL_SELECTION="$2"; shift 2 ;;
        --seed)                SEED="$2"; shift 2 ;;
        --ckpt)                CKPT_SPLIT="$2"; shift 2 ;;
        # optional argument: --shared-assets [DIR]
        --shared-assets)
            if [[ $# -ge 2 && "$2" != --* ]]; then SHARED_ASSETS="$2"; shift 2
            else SHARED_ASSETS="$SHARED_DEFAULT"; shift; fi ;;
        --dry-run)             DRY_RUN=1; shift ;;
        --include-incomplete)  INCLUDE_INCOMPLETE=1; shift ;;
        --include-b1)          INCLUDE_B1=1; shift ;;
        --keep-grids)          KEEP_GRIDS=1; shift ;;
        -h|--help)             usage 0 ;;
        *) echo "unknown option: $1" >&2; usage 1 ;;
    esac
done

METHODS="${METHODS:-$METHODS_DEFAULT}"

if [[ -n "$SHARED_ASSETS" && ! -d "$SHARED_ASSETS" ]]; then
    echo "shared asset folder does not exist: $SHARED_ASSETS" >&2
    exit 1
fi

# --- which detectors ---------------------------------------------------------
mapfile -t ALL_CONFIGS < <(
    ls training/config/detector/*.yaml \
        | xargs -n1 basename | sed 's/\.yaml$//' \
        | grep -vE '^resnet34_bcos_v2_minimal$' \
        | sort
)

config_family() {
    case "$1" in
        resnet34*) echo resnet   ;;
        xception*) echo xception ;;
        vit*)      echo vit      ;;
        convnext*) echo convnext ;;
        *)         echo other    ;;
    esac
}

WANTED=""
for f in $FAMILIES; do
    case "$f" in
        resnet|resnet34)  WANTED="$WANTED resnet"   ;;
        xception)         WANTED="$WANTED xception" ;;
        vit)              WANTED="$WANTED vit"      ;;
        convnext)         WANTED="$WANTED convnext" ;;
        all)              WANTED=""; break          ;;
        *) echo "unknown family: $f (want: resnet xception vit convnext all)" >&2; exit 1 ;;
    esac
done

CONFIGS=()
for cfg in "${ALL_CONFIGS[@]}"; do
    fam="$(config_family "$cfg")"
    [[ -z "$WANTED" || " $WANTED " == *" $fam "* ]] || continue
    # b=1 makes a B-cos layer plain linear: no alignment pressure, and every
    # trained b1 model collapsed to a CONSTANT classifier -- measured accuracy
    # 0.7999 on FF++, exactly the always-predict-fake baseline (4479 real /
    # 17909 fake), AUC 0.54. It therefore classifies ZERO reals correctly, the
    # grid pool has no real cells, and every dataset yields 0 grids after a full
    # (expensive) ranking pass. Excluded by default; --include-b1 to override.
    # Matches b1/b_1 only at the END so b1_25 and b1_75 are untouched.
    if [[ $INCLUDE_B1 -eq 0 && "$cfg" =~ _b_?1$ ]]; then
        echo "SKIP  $cfg  -- b=1 is linear and collapses to a constant classifier (--include-b1 to force)"
        continue
    fi
    CONFIGS+=("$cfg")
done
[[ ${#CONFIGS[@]} -eq 0 ]] && { echo "no configs match:$WANTED" >&2; exit 1; }

# config name -> (run-dir prefix, normalised b-token, raw b-token)
run_tokens() {
    local cfg="$1" family raw norm
    case "$cfg" in
        xception_bcos*)    family="xception_bcos_detector" ;;
        xception)          family="xception" ;;
        vit_bcos*)         family="vit_bcos" ;;
        vit)               family="vit" ;;
        convnext_bcos*)    family="convnext_bcos" ;;
        convnext)          family="convnext" ;;
        resnet34_bcos_v2*) family="resnet34_bcos_v2" ;;
        resnet34)          family="resnet34" ;;
        *)                 family="$cfg" ;;
    esac
    raw=""; norm=""
    if [[ "$cfg" =~ (b_?[0-9]+(_[0-9]+)?)$ ]]; then
        raw="${BASH_REMATCH[1]}"; raw="${raw/b_/b}"; norm="$raw"
        [[ "$norm" =~ ^b[0-9]+$ ]] && norm="${norm}_0"
    fi
    echo "$family|$norm|$raw"
}

find_run_dir() {
    local family="$1" norm="$2" raw="$3" d name
    for d in $(ls -1dt logs/training/*/ 2>/dev/null); do
        name="$(basename "$d")"
        [[ -f "$d/$CKPT_SPLIT/avg/ckpt_best.pth" ]] || continue
        [[ "$name" == "$family"_* ]] || continue
        if [[ -n "$norm" ]]; then
            [[ "$name" == *"_${norm}_"* || "$name" == *"_${norm}" || "$name" == *"_${raw}" ]] || continue
        else
            [[ "$name" == *_bcos_* ]] && continue
        fi
        echo "$d"; return 0
    done
    return 1
}

is_still_training() {
    local d="$1"
    [[ -f "$d/training.log" ]] || return 1
    [[ -n "$(find "$d/training.log" -mmin "-$RUNNING_THRESHOLD_MIN" 2>/dev/null)" ]]
}

n_ds=$(wc -w <<< "$DATASETS")
MAX_GRIDS=$(( GRIDS_PER_DATASET * n_ds ))
[[ -n "$SHARED_ASSETS" ]] && MAX_GRIDS=$(ls "$SHARED_ASSETS"/*.pt 2>/dev/null | wc -l)

STAMP="$(date +%Y-%m-%d-%H-%M-%S)"
OUT_DIR="results/eval/xai/$STAMP"
LOG_DIR="$OUT_DIR/logs"
[[ $DRY_RUN -eq 0 ]] && mkdir -p "$LOG_DIR"

# Shared config overrides. An ARRAY, not a string: these values contain spaces
# and unquoted expansion would word-split them into bogus argv entries.
SETS=(
  --set "test_dataset=[$(sed 's/ /, /g' <<< "$DATASETS")]"
  --set "frame_num={train: 32, test: $FRAME_NUM, val: 32}"
  --set "grid_split=3"
  --set "max_grids=$MAX_GRIDS"
  --set "overwrite=false"
  --set "quantitativ=false"
  --set "threshold_steps=$THRESHOLD_STEPS"
  # The rendered overlays are ~22 MB per grid per threshold and nothing reads
  # them; without this a 320-grid sweep writes ~7 GB per method.
  --set "store_images=false"
)

# Mean weighted/unweighted score per threshold, straight out of the run's pickle.
print_scores() {   # results-root
    "$PYTHON" - "$1" <<'PY'
import os, pickle, sys, numpy as np
root = sys.argv[1]
hit = None
for dirpath, _, files in os.walk(root):
    if "overall_by_threshold.pkl" in files:
        hit = os.path.join(dirpath, "overall_by_threshold.pkl"); break
if not hit:
    print("  (no results pickle found)"); raise SystemExit
d = pickle.load(open(hit, "rb"))
def key(t): return (1, 0) if isinstance(t, str) else (0, t)
print(f"  {'threshold':>10} {'weighted':>10} {'unweighted':>12} {'n':>6}")
for t in sorted(d, key=key):
    w = np.mean(d[t]["weighted_localization_score"])
    u = np.mean(d[t]["unweighted_localization_score"])
    print(f"  {str(t):>10} {w:>10.4f} {u:>12.4f} {len(d[t]['weighted_localization_score']):>6}")
PY
}

echo "repo        : $REPO_ROOT"
echo "python      : $PYTHON"
echo "checkpoint  : <run>/$CKPT_SPLIT/avg/ckpt_best.pth"
echo "datasets    : $n_ds ($(cut -c1-80 <<< "$DATASETS")$([[ $n_ds -gt 6 ]] && echo ' ...'))"
echo "families    : ${WANTED:-all}"
echo "methods     : $METHODS (+ bcos on b-cos models)"
if [[ -n "$SHARED_ASSETS" ]]; then
    echo "grids       : SHARED assets, $MAX_GRIDS grids <- $SHARED_ASSETS"
else
    echo "grids       : per-model confidence selection, ONE RUN PER DATASET,"
    echo "              $GRIDS_PER_DATASET grids each (=> up to $MAX_GRIDS over $n_ds datasets)"
    echo "              (frame_num test=$FRAME_NUM, reals=$REAL_SELECTION, seed=$SEED, cells stay within one dataset)"
fi
echo "thresholds  : 0.0..1.0 in $THRESHOLD_STEPS steps + top-k"
echo "detectors   : ${#CONFIGS[@]} candidates"
[[ $DRY_RUN -eq 0 ]] && echo "output      : $OUT_DIR"
echo

declare -a SUMMARY=()
for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r family norm raw <<< "$(run_tokens "$cfg")"
    if ! run_dir="$(find_run_dir "$family" "$norm" "$raw")"; then
        echo "SKIP  $cfg  -- no $CKPT_SPLIT checkpoint (family=$family b=${norm:-none})"
        SUMMARY+=("SKIP(no-ckpt)   $cfg"); continue
    fi
    run_dir="${run_dir%/}"
    weights="$run_dir/$CKPT_SPLIT/avg/ckpt_best.pth"

    if is_still_training "$run_dir" && [[ $INCLUDE_INCOMPLETE -eq 0 ]]; then
        echo "SKIP  $cfg  -- $(basename "$run_dir") still training (--include-incomplete to override)"
        SUMMARY+=("SKIP(training)  $cfg"); continue
    fi

    # 'bcos' explanations need a b-cos model; everything else is architecture-agnostic.
    cfg_methods="$METHODS"
    [[ "$cfg" == *bcos* ]] && cfg_methods="bcos $METHODS"

    model_out="$OUT_DIR/$cfg"

    # --- grids: shared assets, or one confidence pass reused by every method ---
    if [[ -n "$SHARED_ASSETS" ]]; then
        grid_dir="$SHARED_ASSETS"
    else
        grid_dir="$model_out/grids/3x3"
        if [[ $DRY_RUN -eq 1 ]]; then
            echo "DRY   $cfg  grids <- confidence selection, $GRIDS_PER_DATASET per dataset"
        else
            # ONE INVOCATION PER DATASET. max_grids is a GLOBAL budget inside
            # GPG_eval: it round-robins over the dataset pools and, once a small
            # pool runs dry, keeps handing the remaining budget to the big ones.
            # A pooled run therefore does NOT give N grids per dataset (measured:
            # asking for 20x16 produced 8 for Celeb-DF-v1 and 33 for DFDC). With a
            # single dataset configured the budget IS the per-dataset quota, so
            # each dataset is built separately and the results are merged.
            echo "GRIDS $cfg  <- $(basename "$run_dir")  ($GRIDS_PER_DATASET/dataset)"
            start=$SECONDS
            mkdir -p "$grid_dir"
            : > "$LOG_DIR/${cfg}__grids.log"
            grid_fail=0
            for ds in $DATASETS; do
                ds_out="$model_out/grids_per_dataset/$ds"
                "$PYTHON" "$GPG" \
                    --model-config "training/config/detector/$cfg.yaml" \
                    --test-config "$TEST_CFG" --weights "$weights" \
                    --split test --selection confidence --real-selection "$REAL_SELECTION" \
                    --dataset-mixing single --grids-only --seed "$SEED" \
                    --batch-size 32 --output-dir "$ds_out" \
                    --set "test_dataset=[$ds]" \
                    --set "frame_num={train: 32, test: $FRAME_NUM, val: 32}" \
                    --set "grid_split=3" --set "max_grids=$GRIDS_PER_DATASET" \
                    --set "overwrite=false" --set "quantitativ=false" \
                    --set "threshold_steps=$THRESHOLD_STEPS" --set "store_images=false" \
                    >> "$LOG_DIR/${cfg}__grids.log" 2>&1
                if [[ $? -ne 0 ]]; then
                    echo "      $ds: FAILED (see ${cfg}__grids.log)"; grid_fail=1; continue
                fi
                # GPG_eval names the grid folder <model_name>_<b_value>/3x3 --
                # discover it rather than reconstructing it (b=2.0 -> '2_0',
                # absent -> 'default', model_name != config name for several
                # families). It also creates a SECOND, empty <model>_<config>/3x3
                # for results, so pick the one that actually holds tensors --
                # taking the first match silently yielded zero grids.
                src=""
                while IFS= read -r cand; do
                    if compgen -G "$cand/*.pt" > /dev/null; then src="$cand"; break; fi
                done < <(find "$ds_out" -type d -name '3x3')
                n=0
                [[ -n "$src" ]] && n=$(ls "$src"/*.pt 2>/dev/null | wc -l)
                # Hardlink into the merged folder (same filesystem, no extra disk).
                # Grid indices restart at 0 per dataset, so prefix to avoid clashes.
                for f in "$src"/*.pt; do
                    [[ -e "$f" ]] && ln -f "$f" "$grid_dir/${ds}__$(basename "$f")"
                done
                echo "      $ds: $n grids$([[ $n -lt $GRIDS_PER_DATASET ]] && echo "  (pool exhausted, wanted $GRIDS_PER_DATASET)")"
            done
            # Merged manifest: file -> source dataset, for the per-dataset breakdown.
            "$PYTHON" - "$model_out" "$grid_dir" <<'PY' >> "$LOG_DIR/${cfg}__grids.log" 2>&1
import json, os, sys, glob
model_out, grid_dir = sys.argv[1], sys.argv[2]
merged = {"grids": [], "note": "one GPG_eval run per dataset; 'pool' is the source dataset"}
for mf in sorted(glob.glob(os.path.join(model_out, "grids_per_dataset", "*", "*", "3x3", "manifest.json"))):
    ds = mf.split(os.sep)[-4]
    man = json.load(open(mf))
    for g in man["grids"]:
        g = dict(g); g["file"] = f"{ds}__{g['file']}"; g["pool"] = ds
        merged["grids"].append(g)
    merged.setdefault("selection", man.get("selection"))
    merged.setdefault("real_selection", man.get("real_selection"))
    merged.setdefault("seed", man.get("seed"))
json.dump(merged, open(os.path.join(grid_dir, "manifest.json"), "w"), indent=1)
print(f"merged manifest: {len(merged['grids'])} grids")
PY
            n_grids=$(ls "$grid_dir"/*.pt 2>/dev/null | wc -l)
            echo "      total $n_grids grids in $((SECONDS-start))s"
            if [[ $n_grids -eq 0 ]]; then
                SUMMARY+=("FAIL(grids)     $cfg"); continue
            fi
            [[ $grid_fail -eq 1 ]] && SUMMARY+=("WARN(grids)     $cfg some datasets failed")
        fi
    fi

    for m in $cfg_methods; do
        tag="${cfg}__${m//+/p}"
        if [[ $DRY_RUN -eq 1 ]]; then
            echo "DRY   $tag"; SUMMARY+=("DRY             $tag"); continue
        fi
        echo "RUN   $tag"
        log="$LOG_DIR/$tag.log"
        start=$SECONDS
        "$PYTHON" "$GPG" \
            --model-config "training/config/detector/$cfg.yaml" \
            --test-config "$TEST_CFG" --weights "$weights" \
            --split test --grid-dir "$grid_dir" --xai-method "$m" \
            --output-dir "$model_out/$m" "${SETS[@]}" > "$log" 2>&1
        rc=$?; dur=$((SECONDS - start))
        if [[ $rc -eq 0 ]]; then
            echo "      done in ${dur}s -> $log"
            # Results go BOTH to the console and to the tail of the run's own log.
            { echo; echo "=== mean scores: $tag ==="; print_scores "$model_out/$m"; } \
                | tee -a "$log"
            SUMMARY+=("OK              $tag (${dur}s)")
        else
            echo "      FAILED rc=$rc after ${dur}s -> $log"
            echo "      $(tail -3 "$log" | tr '\n' ' ')"
            SUMMARY+=("FAIL rc=$rc      $tag")
        fi
    done

    # This model's methods are done and nothing else reads its tensors. Drop them
    # (7.1 MB each) but keep manifest.json -- it records every source image, cell
    # position and the seed, so the grids regenerate exactly.
    if [[ $DRY_RUN -eq 0 && $KEEP_GRIDS -eq 0 && -z "$SHARED_ASSETS" ]]; then
        freed=$(du -sh "$model_out/grids" 2>/dev/null | cut -f1)
        find "$model_out" -name '*.pt' -delete 2>/dev/null
        find "$model_out/grids_per_dataset" -type d -empty -delete 2>/dev/null
        echo "      freed $freed of grid tensors (manifest kept; --keep-grids to retain)"
    fi
done

echo
echo "================ summary ================"
printf '%s\n' "${SUMMARY[@]}"

if [[ $DRY_RUN -eq 0 ]]; then
    # One readable file with everything the report needs, instead of numbers
    # buried in per-run pickles.
    echo
    if "$PYTHON" scripts/xai_report.py "$OUT_DIR" -o "$OUT_DIR/RESULTS.txt" > /dev/null 2>"$LOG_DIR/report.err"; then
        echo "RESULTS   -> $OUT_DIR/RESULTS.txt"
    else
        echo "report generation failed; see $LOG_DIR/report.err"
    fi
    echo "results in $OUT_DIR"
    echo "logs in    $LOG_DIR"
fi
