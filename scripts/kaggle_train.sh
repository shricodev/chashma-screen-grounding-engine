#!/usr/bin/env bash
# scripts/kaggle_train.sh push|status|pull|watch
# needs KAGGLE_TOKEN in env or .env
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

stage_data() {
  rm -rf "$STAGE/data" && mkdir -p "$STAGE/data/configs" "$STAGE/data/captures"
  uv build --wheel -q -o "$STAGE/data"
  cp -r data/dataset "$STAGE/data/dataset"
  cp configs/train_256m.yaml "$STAGE/data/configs/"
  (cd data/captures && find . -name '*.png' -exec cp --parents {} "$STAGE/data/captures/" \;)
  cat > "$STAGE/data/dataset-metadata.json" <<EOF
{"title": "chashma-data", "id": "$DATASET", "licenses": [{"name": "other"}]}
EOF
}

wait_dataset_ready() {
  for _ in $(seq 1 60); do
    status="$(kaggle datasets status "$DATASET" 2>/dev/null || true)"
    [[ "$status" == *ready* ]] && return 0
    sleep 10
  done
  echo "dataset $DATASET not ready after 10 minutes" >&2
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
  cp scripts/kaggle/kernel.py "$STAGE/kernel/"
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
  [[ -f "$STAGE/output/chashma-256m/model.safetensors" ]] || {
    echo "no checkpoint in kernel output, see $STAGE/output" >&2
    exit 1
  }
  rm -rf checkpoints/chashma-256m && mkdir -p checkpoints
  mv "$STAGE/output/chashma-256m" checkpoints/chashma-256m
  echo "checkpoint -> checkpoints/chashma-256m"
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
