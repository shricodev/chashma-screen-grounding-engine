#!/usr/bin/env bash
# scripts/kaggle_train.sh push|status|pull|watch
# needs KAGGLE_TOKEN in env or .env. CONFIG=configs/x.yaml to pick a config
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="$ROOT/runs/kaggle"
cd "$ROOT"

if [[ -z "${KAGGLE_API_TOKEN:-}" ]]; then
  KAGGLE_API_TOKEN="${KAGGLE_TOKEN:-$(grep -E '^KAGGLE_TOKEN=' .env 2>/dev/null | cut -d= -f2- | tr -d "\"'")}"
fi
export KAGGLE_API_TOKEN
[[ -n "$KAGGLE_API_TOKEN" ]] || { echo "set KAGGLE_TOKEN in .env" >&2; exit 1; }

kaggle() { uvx -q --from kaggle --with six kaggle "$@"; }

USER_NAME="${KAGGLE_USER:-$(kaggle config view | awk '/username:/ {print $3}')}"
DATASET="$USER_NAME/chashma-data"
KERNEL="$USER_NAME/chashma-train"
STAMP="$(git rev-parse --short HEAD)-$(date +%s)"
CONFIG="${CONFIG:-configs/train_qwen3vl_2b.yaml}"
RUN_NAME="$(basename "$(awk '/^output_dir:/ {print $2}' "$CONFIG")")"

stage_data() {
  rm -rf "$STAGE/data" && mkdir -p "$STAGE/data/configs" "$STAGE/data/captures"
  uv build --wheel -q -o "$STAGE/data"
  cp -r data/dataset "$STAGE/data/dataset"
  cp "$CONFIG" "$STAGE/data/configs/train.yaml"
  touch "$STAGE/data/.stamp-$STAMP.txt"
  (cd data/captures && find . -name '*.png' -exec cp --parents {} "$STAGE/data/captures/" \;)
  cat > "$STAGE/data/dataset-metadata.json" <<EOF
{"title": "chashma-data", "id": "$DATASET", "licenses": [{"name": "other"}]}
EOF
}

# kaggle says ready even for the old version sometimes, so wait for our stamp
wait_dataset_ready() {
  for _ in $(seq 1 120); do
    kaggle datasets files "$DATASET" 2>/dev/null | grep -q ".stamp-$STAMP" && return 0
    sleep 15
  done
  echo "dataset version .stamp-$STAMP not live after 30 minutes" >&2
  return 1
}

push() {
  stage_data
  if kaggle datasets status "$DATASET" >/dev/null 2>&1; then
    kaggle datasets version -p "$STAGE/data" -m "update $(date +%F)" --dir-mode zip
  else
    kaggle datasets create -p "$STAGE/data" --dir-mode zip
  fi
  wait_dataset_ready

  rm -rf "$STAGE/kernel" && mkdir -p "$STAGE/kernel"
  sed "s/__STAMP__/$STAMP/" scripts/kaggle/kernel.py > "$STAGE/kernel/kernel.py"
  cat > "$STAGE/kernel/kernel-metadata.json" <<EOF
{
  "id": "$KERNEL",
  "title": "chashma-train",
  "code_file": "kernel.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": ["$DATASET"],
  "competition_sources": [],
  "kernel_sources": []
}
EOF
  kaggle kernels push -p "$STAGE/kernel" --accelerator NvidiaTeslaT4
}

status() { kaggle kernels status "$KERNEL"; }

pull() {
  rm -rf "$STAGE/output" && mkdir -p "$STAGE/output"
  kaggle kernels output "$KERNEL" -p "$STAGE/output"
  [[ -f "$STAGE/output/$RUN_NAME/chashma.json" ]] || {
    echo "no checkpoint in kernel output, see $STAGE/output" >&2
    exit 1
  }
  rm -rf "checkpoints/$RUN_NAME" && mkdir -p checkpoints
  mv "$STAGE/output/$RUN_NAME" "checkpoints/$RUN_NAME"
  echo "checkpoint -> checkpoints/$RUN_NAME"
}

watch() {
  while true; do
    s="$(status 2>&1 || true)"
    echo "$(date +%T) $s"
    case "$s" in
      *COMPLETE*|*complete*) pull; return 0 ;;
      *ERROR*|*error*|*CANCEL*|*cancel*)
        mkdir -p "$STAGE/output" && kaggle kernels output "$KERNEL" -p "$STAGE/output" || true
        return 1 ;;
    esac
    sleep 120
  done
}

"${1:-push}"
