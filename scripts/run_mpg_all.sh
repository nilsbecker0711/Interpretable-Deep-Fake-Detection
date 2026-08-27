#!/usr/bin/env bash
# Run the MASK Pointing Game for every trained detector x every XAI method.
# Thin driver around notebooks/Linus/GridPointingGame/MPG_eval.py -- the same
# relationship run_xai_all.sh has to GPG_eval.py, and the same flags.
#
#   scripts/run_mpg_all.sh                            # all families, default datasets
#   scripts/run_mpg_all.sh --family resnet            # only the resnet34 pair
#   scripts/run_mpg_all.sh --family "vit convnext"    # two families, one argument
#   scripts/run_mpg_all.sh --datasets "FaceForensics++"
#   scripts/run_mpg_all.sh --methods "gradcam grad++" # default: all five (+bcos)
#   scripts/run_mpg_all.sh --num-images 250           # per dataset, default 500
#   scripts/run_mpg_all.sh --shared-assets            # reuse canonical image lists
#   scripts/run_mpg_all.sh --dry-run
#
# DATASETS: the MPG needs GROUND-TRUTH MANIPULATION MASKS, and only the
# FaceForensics++ family ships them. Verified over every dataset json: FF-DF
# (4473), FF-F2F (4480), FF-FS (4477), FF-NT (4479) and FaceForensics++ (17909,
# the union of the four) have a mask per fake frame; Celeb-DF, DFDC(P), UADFV
# and every DF40 subset have ZERO. Asking for one of those yields no scoreable
# image, so the default is the four manipulation types -- which is also the
# informative split, localization quality per manipulation family.
#
# PROTOCOL: unlike the GPG's confidence selection, the MPG image list is drawn
# MODEL-FREE (seeded random over that dataset's fake frames with masks) and
# selection then happens BY PATH. One list per dataset is built once and reused
# by every model and method, so both the method axis AND the architecture axis
# are matched -- no per-model grids, no confounded comparison.
#
# Every run sweeps thresholds 0.0..1.0 plus the top-k variant; results land in
# <out>/RESULTS.txt via scripts/mpg_report.py.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

PYTHON="${PYTHON:-$HOME/miniconda3/envs/bcos/bin/python}"
MPG="notebooks/Linus/GridPointingGame/MPG_eval.py"
TEST_CFG="training/config/test_config.yaml"

CKPT_SPLIT="val"
DRY_RUN=0
INCLUDE_INCOMPLETE=0
INCLUDE_B1=0          # b=1 is linear and collapses to a constant classifier
FAMILIES=""
NUM_IMAGES=500        # fake images per dataset
THRESHOLD_STEPS=10
SEED=32
FRAME_NUM=32          # preprocessing stored 32 frames per video; 32 = use them all
BATCH_SIZE=12
SHARED_ASSETS=0       # 1 = reuse results/MPG_assets/shared_random/<ds>_test/images.json
RUNNING_THRESHOLD_MIN=30

METHODS_DEFAULT="gradcam xgrad grad++ layergrad ig lime"
METHODS=""

# Only the FF++ family has masks (see header). FaceForensics++ itself is the
# union of these four and can be requested explicitly for a pooled number.
DATASETS_DEFAULT="FF-DF FF-F2F FF-FS FF-NT"
DATASETS="$DATASETS_DEFAULT"

usage() {
    sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --datasets)            DATASETS="$2"; shift 2 ;;
        --family|--families)   FAMILIES="$FAMILIES ${2//,/ }"; shift 2 ;;
        --methods)             METHODS="$METHODS ${2//,/ }"; shift 2 ;;
        --num-images)          NUM_IMAGES="$2"; shift 2 ;;
        --threshold-steps)     THRESHOLD_STEPS="$2"; shift 2 ;;
        --frame-num)           FRAME_NUM="$2"; shift 2 ;;
        --batch-size)          BATCH_SIZE="$2"; shift 2 ;;
        --seed)                SEED="$2"; shift 2 ;;
        --ckpt)                CKPT_SPLIT="$2"; shift 2 ;;
        --shared-assets)       SHARED_ASSETS=1; shift ;;
        --dry-run)             DRY_RUN=1; shift ;;
        --include-incomplete)  INCLUDE_INCOMPLETE=1; shift ;;
        --include-b1)          INCLUDE_B1=1; shift ;;
        -h|--help)             usage 0 ;;
        *) echo "unknown option: $1" >&2; usage 1 ;;
    esac
done

METHODS="${METHODS:-$METHODS_DEFAULT}"

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
    # b=1 makes a B-cos layer plain linear; both trained b1 models collapsed to a
    # constant classifier (FF++ accuracy 0.7999 == the always-fake baseline).
    # Matched at the END only, so b1_25 / b1_75 are untouched.
    if [[ $INCLUDE_B1 -eq 0 && "$cfg" =~ _b_?1$ ]]; then
        echo "SKIP  $cfg  -- b=1 collapses to a constant classifier (--include-b1 to force)"
        continue
    fi
    CONFIGS+=("$cfg")
done
[[ ${#CONFIGS[@]} -eq 0 ]] && { echo "no configs match:$WANTED" >&2; exit 1; }

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

STAMP="$(date +%Y-%m-%d-%H-%M-%S)"
OUT_DIR="results/eval/mpg/$STAMP"
LOG_DIR="$OUT_DIR/logs"
LIST_DIR="$OUT_DIR/image_lists"
[[ $DRY_RUN -eq 0 ]] && mkdir -p "$LOG_DIR" "$LIST_DIR"

# Shared config overrides. An ARRAY, not a string: values contain spaces and
# unquoted expansion would word-split them into bogus argv entries.
common_sets() {   # $1 = dataset
    SETS=(
      --set "test_dataset=[$1]"
      --set "frame_num={train: 32, test: $FRAME_NUM, val: 32}"
      # Detector yamls ship with_mask:false; the MPG cannot score without the
      # ground-truth manipulation mask, so it must be switched on here.
      --set "with_mask=true"
      --set "mask_resolution=256"
      --set "max_images=$NUM_IMAGES"
      --set "overwrite=true"
      --set "quantitativ=false"
      --set "threshold_steps=$THRESHOLD_STEPS"
      # MPG_eval requires xai_method unconditionally -- even for --images-only,
      # which builds the list model-free and never uses it. Placeholder; every
      # real run overrides it with --xai-method, which takes precedence.
      --set "xai_method=gradcam"
    )
}

list_for() {   # $1 = dataset -> path of the image-list asset
    if [[ $SHARED_ASSETS -eq 1 ]]; then
        echo "results/MPG_assets/shared_random/${1}_test/images.json"
    else
        echo "$LIST_DIR/${1}_images.json"
    fi
}

n_ds=$(wc -w <<< "$DATASETS")
echo "repo        : $REPO_ROOT"
echo "python      : $PYTHON"
echo "checkpoint  : <run>/$CKPT_SPLIT/avg/ckpt_best.pth"
echo "datasets    : $n_ds ($DATASETS)"
echo "families    : ${WANTED:-all}"
echo "methods     : $METHODS (+ bcos on b-cos models)"
echo "images      : $NUM_IMAGES per dataset, model-free seeded draw (seed=$SEED)"
echo "              -> the SAME images for every model and method"
echo "thresholds  : 0.0..1.0 in $THRESHOLD_STEPS steps + top-k"
echo "detectors   : ${#CONFIGS[@]} candidates"
[[ $DRY_RUN -eq 0 ]] && echo "output      : $OUT_DIR"
echo

# --- image lists: built ONCE, model-free, reused by everything ---------------
if [[ $DRY_RUN -eq 0 ]]; then
    for ds in $DATASETS; do
        lst="$(list_for "$ds")"
        if [[ -f "$lst" ]]; then
            echo "LIST  $ds  <- existing $(basename "$lst") ($( "$PYTHON" -c "import json,sys;print(len(json.load(open('$lst'))['images']))" 2>/dev/null || echo '?') images)"
            continue
        fi
        echo "LIST  $ds  building $NUM_IMAGES-image list"
        common_sets "$ds"
        # A detector yaml is required even for --images-only (it carries
        # resolution / dataset_type); any model of the right channel count does,
        # since no weights are loaded. Use the first selected config.
        "$PYTHON" "$MPG" \
            --model-config "training/config/detector/${CONFIGS[0]}.yaml" \
            --test-config "$TEST_CFG" --images-only \
            --num-images "$NUM_IMAGES" --seed "$SEED" \
            --image-list "$lst" --batch-size "$BATCH_SIZE" \
            --output-dir "$LIST_DIR" \
            "${SETS[@]}" > "$LOG_DIR/list_${ds}.log" 2>&1
        if [[ $? -ne 0 || ! -f "$lst" ]]; then
            echo "      FAILED -> $LOG_DIR/list_${ds}.log"
            echo "      $(tail -3 "$LOG_DIR/list_${ds}.log" | tr '\n' ' ')"
        else
            echo "      $( "$PYTHON" -c "import json;print(len(json.load(open('$lst'))['images']))" ) images"
        fi
    done
    echo
fi

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

    # 'bcos' explanations call backbone.explain(), which only b-cos backbones
    # have; drop it for standard models even when asked for explicitly, and add
    # it for b-cos models when it was not listed.
    if [[ "$cfg" == *bcos* ]]; then
        cfg_methods="$METHODS"
        [[ " $METHODS " == *" bcos "* ]] || cfg_methods="bcos $METHODS"
    else
        cfg_methods=""
        for m in $METHODS; do
            [[ "$m" == bcos ]] && continue
            cfg_methods="$cfg_methods $m"
        done
    fi
    [[ -z "${cfg_methods// }" ]] && { echo "SKIP  $cfg -- no applicable methods"; continue; }

    for m in $cfg_methods; do
        for ds in $DATASETS; do
            tag="${cfg}__${m//+/p}__${ds}"
            if [[ $DRY_RUN -eq 1 ]]; then
                echo "DRY   $tag"; SUMMARY+=("DRY             $tag"); continue
            fi
            lst="$(list_for "$ds")"
            if [[ ! -f "$lst" ]]; then
                echo "SKIP  $tag -- no image list"
                SUMMARY+=("SKIP(no-list)   $tag"); continue
            fi
            echo "RUN   $tag"
            log="$LOG_DIR/$tag.log"
            common_sets "$ds"
            start=$SECONDS
            "$PYTHON" "$MPG" \
                --model-config "training/config/detector/$cfg.yaml" \
                --test-config "$TEST_CFG" --weights "$weights" \
                --xai-method "$m" --image-list "$lst" \
                --batch-size "$BATCH_SIZE" \
                --output-dir "$OUT_DIR/$cfg/$m/$ds" \
                "${SETS[@]}" > "$log" 2>&1
            rc=$?; dur=$((SECONDS - start))
            if [[ $rc -eq 0 ]]; then
                echo "      done in ${dur}s -> $log"
                SUMMARY+=("OK              $tag (${dur}s)")
            else
                echo "      FAILED rc=$rc after ${dur}s -> $log"
                echo "      $(tail -3 "$log" | tr '\n' ' ')"
                SUMMARY+=("FAIL rc=$rc      $tag")
            fi
        done
    done
done

echo
echo "================ summary ================"
printf '%s\n' "${SUMMARY[@]}"

if [[ $DRY_RUN -eq 0 ]]; then
    echo
    if "$PYTHON" scripts/mpg_report.py "$OUT_DIR" -o "$OUT_DIR/RESULTS.txt" \
            > /dev/null 2>"$LOG_DIR/report.err"; then
        echo "RESULTS   -> $OUT_DIR/RESULTS.txt"
    else
        echo "report generation failed; see $LOG_DIR/report.err"
    fi
    echo "results in $OUT_DIR"
    echo "logs in    $LOG_DIR"
fi
