#!/usr/bin/env bash
# Nightly run: pull latest code, queue tonight's jobs, run them, write and commit the report.
# Run by systemd/lab-nightly.timer. Safe to run by hand: `bash jobs/nightly.sh`.
set -euo pipefail

LAB_ROOT="${LAB_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
export LAB_ROOT
cd "$LAB_ROOT"

STOP_AT="${LAB_STOP_AT:-08:30}"          # stop starting new jobs after this local time
PROJECT_REPOS="${LAB_PROJECT_REPOS:-../scam-radar ../urdu-jailbreak-bench ../cricket-shot-coach ../urdu-token-tax}"
PY="${LAB_PYTHON:-python3}"
MIN_FREE_GB="${LAB_MIN_FREE_GB:-50}"

log() { echo "[$(date -Is)] $*"; }

pull() {
  local dir="$1"
  if [ -d "$dir/.git" ]; then
    log "pull $dir"
    git -C "$dir" pull --ff-only --quiet || log "WARN: could not fast-forward $dir (local changes?)"
  fi
}

pull "$LAB_ROOT"
for repo in $PROJECT_REPOS; do pull "$LAB_ROOT/$repo"; done

shopt -s nullglob
jobs=(jobs/nightly.d/*.json)
if [ ${#jobs[@]} -gt 0 ]; then
  "$PY" -m lab enqueue "${jobs[@]}"
else
  log "no jobs in jobs/nightly.d/"
fi

# The always-on worker service may already be draining the queue; if so, wait for it.
if systemctl --user is-active --quiet lab-worker.service 2>/dev/null; then
  log "lab-worker.service is running; waiting for the queue to empty or $STOP_AT"
  while [ -n "$(ls jobs/queue/*.json jobs/running/*.json 2>/dev/null)" ]; do
    [ "$(date +%H:%M)" \> "$STOP_AT" ] && [ "$(date +%H:%M)" \< "12:00" ] && break
    sleep 60
  done
else
  "$PY" -m lab worker --drain --until "$STOP_AT" --min-free-gb "$MIN_FREE_GB" || true
fi

REPORT="$("$PY" -m lab report)"
log "report: $REPORT"

if git -C "$LAB_ROOT" remote get-url origin >/dev/null 2>&1; then
  git add "$REPORT"
  git commit --quiet -m "report: $(basename "$REPORT" .md)" || true
  git push --quiet || log "WARN: push failed; report kept locally"
fi

"$PY" - <<'EOF'
import os, pathlib, sys
sys.path.insert(0, os.environ["LAB_ROOT"])
from lab.notify import notify
r = sorted(pathlib.Path(os.environ["LAB_ROOT"], "reports").glob("*.md"))[-1]
first = next((l for l in r.read_text().splitlines() if l.startswith("**")), "report written")
notify("nightly report", first.strip("*"), tags="memo")
EOF
