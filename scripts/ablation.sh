#!/usr/bin/env bash
# Softness ablation: hard vs soft triangles across three sequences.
#
#   ./scripts/ablation.sh              # six runs, no video
#   ./scripts/ablation.sh --video      # ... and record each one (much slower)
#   ./scripts/ablation.sh --npz        # ... and keep the per-keyframe .npz dumps
#   ./scripts/ablation.sh --max_kf 20  # short smoke test first
#
# Every run gets its own output directory, because panels, patch npz files,
# video frames and the timing CSV all land in --out_dir and would otherwise
# overwrite each other. Full terminal output is teed per run, and a combined
# digest of the headline numbers is written at the end.
set -uo pipefail          # deliberately not -e: one failed run must not kill the sweep

cd "$(dirname "$0")/.."
PY=${PY:-python}
# BSD date has no -Is; fall back so the script runs on macOS too
now () { date -Is 2>/dev/null || date "+%Y-%m-%dT%H:%M:%S%z"; }
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=${OUT:-runs/ablation_$STAMP}
mkdir -p "$OUT"

VIDEO=()
NPZ=( --no_patch_npz )   # ~1MB per keyframe per run; --npz to keep them
EXTRA=()
for a in "$@"; do
  case "$a" in
    --video) VIDEO=( --video --video_mode rgb --video_fov_scale 1.5 ) ;;
    --npz)   NPZ=() ;;
    *)       EXTRA+=( "$a" ) ;;
  esac
done

# Everything held fixed across the sweep. Only sigma varies.
COMMON=( --occlusion_tol 0.50 --occlusion_min_area 100 --min_region_frac 0.01
         --opacity 0.28 --iters_per_second 1000 --age_max_patches 100
         --tail_seconds 10 --evaluate --eval_holdout 6 )

# office0's ground-truth depth calibration refuses (its record indices do not
# match the dataset frame numbers), and freiburg1_desk has no _depth folder at
# all, so it falls back on its own. Only freiburg1_room has usable GT depth.
declare -a SEQS=( office0 freiburg1_desk freiburg1_room )
seq_flags () {
  case "$1" in
    office0) echo "--no_gt_depth" ;;
    *)       echo "" ;;
  esac
}

# The two conditions. "soft" is the TS+ setting annealed through the tail;
# "hard" is near-opaque triangles with no anneal.
cond_flags () {
  case "$1" in
    soft) echo "--eval_lpips --sigma 1.0 --sigma_final 0.0001 --sigma_anneal_seconds 10" ;;
    hard) echo "--eval_lpips --sigma 0.001 --sigma_anneal_seconds 0" ;;
  esac
}

echo "sweep -> $OUT"
for ds in "${SEQS[@]}"; do
  for cond in soft hard; do
    tag="${ds}_${cond}"
    run_out="$OUT/$tag"
    log="$OUT/${tag}.log"
    mkdir -p "$run_out"

    cmd=( "$PY" mapper/harness.py
          --records_dir "data/${ds}_records"
          --out_dir "$run_out"
          "${COMMON[@]}" ${VIDEO[@]+"${VIDEO[@]}"} ${NPZ[@]+"${NPZ[@]}"}
          $(seq_flags "$ds") $(cond_flags "$cond")
          ${EXTRA[@]+"${EXTRA[@]}"} )

    {
      echo "### $tag"
      echo "### started $(now)"
      echo "### ${cmd[*]}"
      echo
    } > "$log"

    echo
    echo "=============================================================="
    echo "  $tag"
    echo "=============================================================="
    # </dev/null matters when detached: a background child that reads from a
    # terminal which has gone away is stopped with SIGTTIN, which looks exactly
    # like the run having silently died.
    "${cmd[@]}" </dev/null 2>&1 | tee -a "$log"
    rc=${PIPESTATUS[0]}
    echo "### exit $rc at $(now)" >> "$log"
    [ "$rc" -ne 0 ] && echo "  !! $tag exited $rc (continuing)"
  done
done

"$PY" scripts/ablation_digest.py "$OUT" | tee "$OUT/summary.txt"
echo
echo "logs, per-run outputs and summary.txt are in $OUT"
