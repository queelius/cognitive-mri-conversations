#!/usr/bin/env bash
# Reproduce all temporal analysis results and figures from the curated data.
#
# Usage (from repo root):
#   bash data/reproduce.sh
#
# Prerequisites:
#   pip install -r code/requirements.txt
#
# Outputs:
#   data/temporal/          — CSV/JSON data files (overwritten)
#   comp-net-2025-journal/paper/figures/temporal/ — PDF/PNG figures (overwritten)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CODE_DIR="$REPO_ROOT/comp-net-2025-journal/code"

echo "============================================================"
echo "  Reproducing temporal analysis from data/"
echo "  Repo root: $REPO_ROOT"
echo "============================================================"

cd "$CODE_DIR"

python run_temporal_analysis.py \
    --embeddings-dir "$REPO_ROOT/data/embeddings" \
    --edges-file "$REPO_ROOT/data/network/edges_user2.0-ai1.0_t0.9.json" \
    --output-dir "$REPO_ROOT/comp-net-2025-journal/paper/figures/temporal" \
    --data-output-dir "$REPO_ROOT/data/temporal"

echo ""
echo "Done. Compare outputs with previous results:"
echo "  diff data/temporal/ <backup>"
echo "  diff comp-net-2025-journal/paper/figures/temporal/ <backup>"
