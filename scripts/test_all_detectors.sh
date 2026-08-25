#!/usr/bin/env bash
# Test every trained detector on a set of datasets.
#
#   scripts/test_all_detectors.sh                 # all families, the default dataset list
#   scripts/test_all_detectors.sh --family resnet # only the resnet34 pair (std + b-cos sweep)
#   scripts/test_all_detectors.sh --family vit --family convnext   # two families
#   scripts/test_all_detectors.sh --family "resnet xception vit"   # same, one argument
#   scripts/test_all_detectors.sh --df40          # all usable DF40 subsets instead
#   scripts/test_all_detectors.sh --datasets "FaceForensics++ Celeb-DF-v2"
#   scripts/test_all_detectors.sh --dry-run       # print what would run, touch nothing
#
# Families: resnet, xception, vit, convnext (default: all). Each covers the
# standard model and its whole b-cos b-value sweep.
#
# One `python training/test.py` invocation per detector, run sequentially. A
# failing model does not stop the sweep; the summary at the end reports each one.
#
# Checkpoints are DISCOVERED, not hardcoded: run directory names are inconsistent
# (b1 vs b1_0, xception_bcos -> xception_bcos_detector, and the resnet runs put the
# b-value AFTER the timestamp), so a model is matched by family + b-token appearing
# anywhere in the directory name, and the newest match wins.
set -uo pipefail

# test.py opens ./training/config/test_config.yaml relative to CWD, so the repo
# root is mandatory -- from anywhere else it dies on a confusing FileNotFoundError.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

PYTHON="${PYTHON:-$HOME/miniconda3/envs/bcos/bin/python}"
CKPT_SPLIT="val"          # val/avg/ckpt_best.pth ; --ckpt test to use the test dir
DRY_RUN=0
INCLUDE_INCOMPLETE=0
FAMILIES=""               # empty = every family
# A run whose training.log changed this recently is assumed to be still training.
RUNNING_THRESHOLD_MIN=30

#DATASETS_DEFAULT="FSAll_cdf FRAll_cdf EFSAll_cdf"
DATASETS_DEFAULT="FaceForensics++ Celeb-DF-v1 Celeb-DF-v2 DFDCP DFDC UADFV FSAll_ff FRAll_ff EFSAll_ff deepfacelab heygen MidJourney whichisreal CollabDiff e4e_ff e4e_cdf FSAll_cdf FRAll_cdf EFSAll_cdf"
DATASETS="$DATASETS_DEFAULT"

# DF40 subsets: everything with a usable test split, minus the classic benchmarks.
# DF40_all/stargan/starganv2/styleclip have EMPTY test splits and would yield 0
# frames; uniface.json/simswap.json were archived (stale paths + wrong labels).
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
        --df40)               DATASETS="$(df40_datasets)"; shift ;;
        --datasets)           DATASETS="$2"; shift 2 ;;
        --family|--families)  FAMILIES="$FAMILIES ${2//,/ }"; shift 2 ;;
        --ckpt)               CKPT_SPLIT="$2"; shift 2 ;;
        --dry-run)            DRY_RUN=1; shift ;;
        --include-incomplete) INCLUDE_INCOMPLETE=1; shift ;;
        -h|--help)            usage 0 ;;
        *) echo "unknown option: $1" >&2; usage 1 ;;
    esac
done

# --- which detectors to test -------------------------------------------------
# Everything in training/config/detector/ except the unused *_minimal config.
mapfile -t ALL_CONFIGS < <(
    ls training/config/detector/*.yaml \
        | xargs -n1 basename | sed 's/\.yaml$//' \
        | grep -vE '^resnet34_bcos_v2_minimal$' \
        | sort
)

# architecture family of a config (the b-cos sweep and its standard twin share one)
config_family() {
    case "$1" in
        resnet34*) echo resnet   ;;
        xception*) echo xception ;;
        vit*)      echo vit      ;;
        convnext*) echo convnext ;;
        *)         echo other    ;;
    esac
}

# normalise user input: 'resnet34' -> 'resnet', etc.
WANTED=""
for f in $FAMILIES; do
    case "$f" in
        resnet|resnet34)   WANTED="$WANTED resnet"   ;;
        xception)          WANTED="$WANTED xception" ;;
        vit)               WANTED="$WANTED vit"      ;;
        convnext)          WANTED="$WANTED convnext" ;;
        all)               WANTED=""; break          ;;
        *) echo "unknown family: $f (want: resnet xception vit convnext all)" >&2; exit 1 ;;
    esac
done

CONFIGS=()
for cfg in "${ALL_CONFIGS[@]}"; do
    fam="$(config_family "$cfg")"
    if [[ -z "$WANTED" ]] || [[ " $WANTED " == *" $fam "* ]]; then
        CONFIGS+=("$cfg")
    fi
done
if [[ ${#CONFIGS[@]} -eq 0 ]]; then
    echo "no detector configs match the requested families:$WANTED" >&2
    exit 1
fi

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

    # Two spellings exist in logs/training/:
    #   convnext/vit/xception runs pad the b-value  -> ..._b1_0_<timestamp>
    #   resnet runs keep it raw and put it LAST     -> ..._<timestamp>_b1
    raw=""; norm=""
    if [[ "$cfg" =~ (b_?[0-9]+(_[0-9]+)?)$ ]]; then
        raw="${BASH_REMATCH[1]}"
        raw="${raw/b_/b}"                       # xception_bcos_b_1 -> b1
        norm="$raw"
        [[ "$norm" =~ ^b[0-9]+$ ]] && norm="${norm}_0"
    fi
    echo "$family|$norm|$raw"
}

# newest run dir matching the family (+ b-token, if any) that holds a checkpoint
find_run_dir() {
    local family="$1" norm="$2" raw="$3" d name
    for d in $(ls -1dt logs/training/*/ 2>/dev/null); do
        name="$(basename "$d")"
        [[ -f "$d/$CKPT_SPLIT/avg/ckpt_best.pth" ]] || continue
        [[ "$name" == "$family"_* ]] || continue
        if [[ -n "$norm" ]]; then
            # padded form may sit anywhere; the RAW form is matched only at the
            # end, otherwise '_b1' would also match '..._b1_25'
            [[ "$name" == *"_${norm}_"* || "$name" == *"_${norm}" || "$name" == *"_${raw}" ]] || continue
        else
            # a bare family (e.g. 'vit') must not match its b-cos sibling
            [[ "$name" == *_bcos_* ]] && continue
        fi
        echo "$d"
        return 0
    done
    return 1
}

is_still_training() {   # training.log touched in the last RUNNING_THRESHOLD_MIN minutes
    local d="$1"
    [[ -f "$d/training.log" ]] || return 1
    [[ -n "$(find "$d/training.log" -mmin "-$RUNNING_THRESHOLD_MIN" 2>/dev/null)" ]]
}

STAMP="$(date +%Y-%m-%d-%H-%M-%S)"
LOG_DIR="logs/testing/$STAMP"
[[ $DRY_RUN -eq 0 ]] && mkdir -p "$LOG_DIR"

n_ds=$(wc -w <<< "$DATASETS")
echo "repo      : $REPO_ROOT"
echo "python    : $PYTHON"
echo "checkpoint: <run>/$CKPT_SPLIT/avg/ckpt_best.pth"
echo "datasets  : $n_ds  ($(cut -c1-90 <<< "$DATASETS")$([[ $n_ds -gt 8 ]] && echo ' ...'))"
echo "families  : ${WANTED:-all}"
echo "detectors : ${#CONFIGS[@]} candidates"
[[ $DRY_RUN -eq 0 ]] && echo "logs      : $LOG_DIR"
echo

declare -a SUMMARY=()
for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r family norm raw <<< "$(run_tokens "$cfg")"
    if ! run_dir="$(find_run_dir "$family" "$norm" "$raw")"; then
        echo "SKIP  $cfg  -- no $CKPT_SPLIT checkpoint found (family=$family b=${norm:-none})"
        SUMMARY+=("SKIP(no-ckpt)  $cfg")
        continue
    fi
    run_dir="${run_dir%/}"
    weights="$run_dir/$CKPT_SPLIT/avg/ckpt_best.pth"

    if is_still_training "$run_dir" && [[ $INCLUDE_INCOMPLETE -eq 0 ]]; then
        echo "SKIP  $cfg  -- $(basename "$run_dir") is still training (--include-incomplete to override)"
        SUMMARY+=("SKIP(training) $cfg")
        continue
    fi

    cmd=("$PYTHON" training/test.py
         --detector_path "training/config/detector/$cfg.yaml"
         --weights_path "$weights"
         --test_dataset $DATASETS)

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "DRY   $cfg"
        printf '        %s\n' "${cmd[*]}"
        SUMMARY+=("DRY            $cfg")
        continue
    fi

    echo "RUN   $cfg  <- $(basename "$run_dir")"
    log="$LOG_DIR/$cfg.log"
    start=$SECONDS
    "${cmd[@]}" > "$log" 2>&1
    rc=$?
    dur=$((SECONDS - start))
    if [[ $rc -eq 0 ]]; then
        echo "      done in ${dur}s -> $log"
        SUMMARY+=("OK             $cfg (${dur}s)")
    else
        echo "      FAILED rc=$rc after ${dur}s -> $log"
        echo "      $(tail -3 "$log" | tr '\n' ' ')"
        SUMMARY+=("FAIL rc=$rc     $cfg")
    fi
done

echo
echo "================ summary ================"
printf '%s\n' "${SUMMARY[@]}"
[[ $DRY_RUN -eq 0 ]] && echo && echo "logs in $LOG_DIR"
