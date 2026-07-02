#!/usr/bin/env bash
# Run the Open ASR Leaderboard WER bench one model per process, then merge +
# splice the README. Running each model in its own `uv run` process means:
#   * no model stacking in one Python process (full VRAM/RAM freed between
#     models via process exit) -- the single-process run OOM-rebooted a 31GB
#     WSL2 box; this isolation makes a repeat freeze recoverable (lose only
#     the model in flight).
#   * a crash on one model doesn't lose the others (each writes its own JSON).
#
# Usage:
#   HF_TOKEN=... bash benchmarks/run_leaderboard_all.sh
#   MODELS="granite,parakeet" ENGINES="starling,stock" NUM_SAMPLES=50 bash benchmarks/run_leaderboard_all.sh
#
# Env knobs: MODELS (comma list), ENGINES (comma list), NUM_SAMPLES (int),
#   DATASETS (comma list), SKIP_RUN=1 to only merge existing per-model JSONs.
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS="${MODELS:-granite,parakeet,moss,qwen3,ark,cohere}"
ENGINES="${ENGINES:-starling,stock}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"
DATASETS="${DATASETS:-}"

OUT_DIR="outputs/leaderboard_per_model"
mkdir -p "$OUT_DIR"

dataset_arg=""
if [ -n "$DATASETS" ]; then dataset_arg="--datasets $DATASETS"; fi

if [ "${SKIP_RUN:-0}" != "1" ]; then
  for m in $(echo "$MODELS" | tr ',' ' '); do
    out="$OUT_DIR/leaderboard_${m}.json"
    echo "=================================================================="
    echo "[driver] model=$m engines=$ENGINES -> $out"
    echo "=================================================================="
    # Run in isolation; allow failure (a crash on one model is logged + skipped
    # so the remaining models still complete). `|| true` keeps the loop alive.
    uv run python benchmarks/bench_leaderboard.py \
      --models "$m" --engines "$ENGINES" \
      --num-samples "$NUM_SAMPLES" $dataset_arg \
      --out "$out" || {
        echo "[driver] WARNING: model $m failed (see above); continuing."
      }
  done
fi

# Merge all per-model JSONs into one table + splice README.
jsons=( "$OUT_DIR"/leaderboard_*.json )
echo "[driver] merging ${#jsons[@]} per-model JSONs -> README"
uv run python benchmarks/bench_leaderboard.py --from-json "${jsons[@]}" --update-readme
echo "[driver] done. WER table spliced into README.md."
