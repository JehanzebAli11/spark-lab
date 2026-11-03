#!/usr/bin/env bash
# Upload a model or dataset folder to a PRIVATE Hugging Face repo. Run by the owner only;
# coding agents must never run this (see CLAUDE.md). Uses HF_WRITE_TOKEN, kept out of
# spark-lab.env so services and agents never see it.
#
#   HF_WRITE_TOKEN=hf_... bash publish/publish.sh model JehanzebAli11/scam-radar-270m ../scam-radar/runs/v1/final
#   HF_WRITE_TOKEN=hf_... bash publish/publish.sh dataset JehanzebAli11/roman-urdu-scam-bench ./release
set -euo pipefail

KIND="${1:?model|dataset}"; REPO="${2:?owner/name}"; DIR="${3:?folder to upload}"
case "$KIND" in model|dataset) ;; *) echo "kind must be model or dataset" >&2; exit 2;; esac
[ -n "${HF_WRITE_TOKEN:-}" ] || { echo "set HF_WRITE_TOKEN for this command only" >&2; exit 2; }
[ -d "$DIR" ] || { echo "no such folder: $DIR" >&2; exit 2; }
case "$(cd "$DIR" && pwd)" in *eval/locked*) echo "refusing: folder is inside eval/locked" >&2; exit 2;; esac
[ -f "$DIR/README.md" ] || { echo "refusing: $DIR has no README.md (model/dataset card)" >&2; exit 2; }

echo "About to upload $DIR to PRIVATE $KIND repo $REPO:"
(cd "$DIR" && find . -type f | head -40)
read -r -p "Type the repo name to confirm: " CONFIRM
[ "$CONFIRM" = "$REPO" ] || { echo "cancelled"; exit 1; }

HF_TOKEN="$HF_WRITE_TOKEN" hf repo create "$REPO" --repo-type "$KIND" --private --exist-ok
HF_TOKEN="$HF_WRITE_TOKEN" hf upload "$REPO" "$DIR" . --repo-type "$KIND" --private
echo "Uploaded privately. Make it public yourself on huggingface.co when the card is ready."
